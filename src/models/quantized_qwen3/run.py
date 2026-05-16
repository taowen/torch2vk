"""Standalone Qwen3 Q4_K_M Vulkan generation.

Run from project root:
    uv run python -m models.quantized_qwen3.run
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from transformers import AutoConfig

from models.hf_cache import resolve_cached_model
from models.quantized_qwen3.dispatch.decode_embed import run_decode_embed
from models.quantized_qwen3.dispatch.decode_layer import run_decode_layer
from models.quantized_qwen3.dispatch.decode_norm import run_decode_norm
from models.quantized_qwen3.dispatch.embed_tokens import run_embed_tokens
from models.quantized_qwen3.dispatch.text_layer import run_text_layer
from models.quantized_qwen3.dispatch.text_norm import run_text_norm
from models.quantized_qwen3.export_gguf import REPO_ID, export_qwen3_q4_k_m_gguf
from models.quantized_qwen3.input_prep import (
    DEFAULT_PROMPT,
    load_qwen3_tokenizer,
    prepare_qwen3_inputs,
)
from models.quantized_qwen3.shaders.lm_head_q6_k_argmax_partial_f16 import (
    LM_HEAD_Q6_K_ARGMAX_PARTIAL_F16,
)
from models.quantized_qwen3.shaders.qwen3_token_select_reduce_chunks_f32 import (
    QWEN3_TOKEN_SELECT_REDUCE_CHUNKS_F32,
)
from models.quantized_qwen3.shaders.qwen3_token_select_reduce_f32 import (
    QWEN3_TOKEN_SELECT_REDUCE_F32,
)
from models.quantized_qwen3.shaders.qwen3_token_store_eos import QWEN3_TOKEN_STORE_EOS
from models.quantized_qwen3.shaders.slice_last_token_f16 import SLICE_LAST_TOKEN_F16
from models.quantized_qwen3.tensors.model import create_model_tensors, model_tensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.replay import ReplayPlan, execute_replay, stage_replay_step_inputs
from torch2vk.runtime.replay_cache_key import (
    build_cached_replay_plan,
    cached_replay_plan,
    replay_cache_namespace,
    source_tree_digest,
)
from torch2vk.runtime.rope_table import ROPE_TABLE_F32
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader


PREFILL_REPLAY_CACHE = "quantized_qwen3_prefill:v4"
DECODE_REPLAY_CACHE = "quantized_qwen3_decode_step:v4"
get_shader = make_shader_loader(
    "models.quantized_qwen3.shaders",
    extra_variants=(ROPE_TABLE_F32,),
)


_REPLAY_SOURCE_DIGEST = source_tree_digest(__file__)


@dataclass(frozen=True, slots=True)
class Qwen3RunResult:
    text: str
    prompt_length: int
    generated_tokens: int
    prefill_elapsed: float
    decode_elapsed: float

    @property
    def decode_ms_per_token(self) -> float:
        if self.generated_tokens <= 1:
            return 0.0
        return self.decode_elapsed / (self.generated_tokens - 1) * 1000.0

    @property
    def prefill_tokens_per_second(self) -> float:
        if self.prefill_elapsed <= 0.0:
            return 0.0
        return self.prompt_length / self.prefill_elapsed


def _require_gpu_output(tensor: LogicalTensor) -> None:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} did not produce a GPU buffer")


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _run_lm_head_select(rt: RuntimeSession, *, x: LogicalTensor) -> None:
    tensors = model_tensors()
    LM_HEAD_Q6_K_ARGMAX_PARTIAL_F16(
        rt,
        x=x,
        weight=tensors.lm_head.p_weight,
        partial_scores=tensors.lm_head_partial_scores,
        partial_tokens=tensors.lm_head_partial_tokens,
    )
    QWEN3_TOKEN_SELECT_REDUCE_CHUNKS_F32(
        rt,
        scores=tensors.lm_head_partial_scores,
        tokens=tensors.lm_head_partial_tokens,
        chunk_scores=tensors.lm_head_chunk_scores,
        chunk_tokens=tensors.lm_head_chunk_tokens,
    )
    QWEN3_TOKEN_SELECT_REDUCE_F32(
        rt,
        partial_scores=tensors.lm_head_chunk_scores,
        partial_tokens=tensors.lm_head_chunk_tokens,
        eos_token_ids=tensors.eos_token_ids,
        next_token=tensors.next_token,
        done=tensors.done,
    )


def _run_prefill(rt: RuntimeSession, *, rope_theta: float) -> None:
    tensors = model_tensors()
    with rt.frame("qwen3.prefill"):
        ROPE_TABLE_F32(
            rt,
            start_position=0,
            theta=rope_theta,
            cos=tensors.prefill_rope.cos,
            sin=tensors.prefill_rope.sin,
        )
        run_embed_tokens(rt)
        for layer_idx in range(len(tensors.text_layers)):
            run_text_layer(rt, layer_idx)
        run_text_norm(rt)
        SLICE_LAST_TOKEN_F16(
            rt,
            x=tensors.text_norm.mul_1,
            output=tensors.prefill_lm_head_input,
        )
        _run_lm_head_select(rt, x=tensors.prefill_lm_head_input)
        QWEN3_TOKEN_STORE_EOS(
            rt,
            next_token=tensors.next_token,
            token_index=0,
            done=tensors.done,
            generated_tokens=tensors.generated_tokens,
            generated_length=tensors.generated_length,
            stopped=tensors.stopped,
        )


def _run_decode_step(
    rt: RuntimeSession,
    *,
    cache_position: int,
    rope_theta: float,
    step: int,
) -> None:
    tensors = model_tensors()
    with rt.frame(f"qwen3.decode.{step:04d}"):
        ROPE_TABLE_F32(
            rt,
            start_position=cache_position,
            theta=rope_theta,
            cos=tensors.decode_rope.cos,
            sin=tensors.decode_rope.sin,
        )
        run_decode_embed(rt)
        for layer_idx in range(len(tensors.decode_layers)):
            run_decode_layer(rt, layer_idx, cache_position=cache_position)
        run_decode_norm(rt)
        _run_lm_head_select(rt, x=tensors.decode_norm.mul_1)
        QWEN3_TOKEN_STORE_EOS(
            rt,
            next_token=tensors.next_token,
            token_index=step + 1,
            done=tensors.done,
            generated_tokens=tensors.generated_tokens,
            generated_length=tensors.generated_length,
            stopped=tensors.stopped,
        )


def _prefill_inputs(
    *,
    input_ids: np.ndarray,
    prompt_length: int,
) -> dict[LogicalTensor, np.ndarray]:
    tensors = model_tensors()
    return {
        tensors.input_ids: input_ids,
        tensors.text_layers[0].cache_position: np.arange(prompt_length, dtype=np.int64),
    }


def _read_selected_token(rt: RuntimeSession, next_token: LogicalTensor) -> int:
    _require_gpu_output(next_token)
    return int(rt.read_request_state(next_token).reshape(-1)[0])


def _build_replay_plan(
    rt: RuntimeSession,
    *,
    name: str,
    frame: str,
    cache_namespace: str,
) -> ReplayPlan:
    return build_cached_replay_plan(
        rt,
        namespace=cache_namespace,
        name=name,
        frame=frame,
        readback_error=f"{name} replay must not use readback slots",
    )


def _cached_replay_plan(
    rt: RuntimeSession,
    *,
    cache_namespace: str,
) -> ReplayPlan | None:
    return cached_replay_plan(rt, namespace=cache_namespace)


def _prefill_replay_cache_namespace(
    *,
    model_dir: Path,
    prompt_length: int,
    max_sequence_length: int,
) -> str:
    return replay_cache_namespace(
        name=PREFILL_REPLAY_CACHE,
        source_digest=_REPLAY_SOURCE_DIGEST,
        model_dir=model_dir,
        shape_key=f"prompt={prompt_length}:max={max_sequence_length}",
    )


def _decode_replay_cache_namespace(
    *,
    model_dir: Path,
    max_sequence_length: int,
) -> str:
    return replay_cache_namespace(
        name=DECODE_REPLAY_CACHE,
        source_digest=_REPLAY_SOURCE_DIGEST,
        model_dir=model_dir,
        shape_key=f"max={max_sequence_length}",
    )


def main(
    *,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 64,
    profile_dir: str | Path | None = None,
) -> Qwen3RunResult:
    if max_new_tokens <= 0 or max_new_tokens > 128:
        raise ValueError(f"max_new_tokens must be in [1, 128], got {max_new_tokens}")

    model_dir = resolve_cached_model(REPO_ID)
    gguf_path = export_qwen3_q4_k_m_gguf(model_dir=model_dir)
    tokenizer = load_qwen3_tokenizer(model_dir)
    prepared = prepare_qwen3_inputs(tokenizer=tokenizer, prompt=prompt)
    config = AutoConfig.from_pretrained(model_dir)
    prompt_length = prepared.prompt_length
    max_sequence_length = _round_up(prompt_length + 128, 64)
    rope_theta = float(getattr(config, "rope_theta", 1_000_000.0))
    eos_token_ids = (int(config.eos_token_id),)
    eos_token_array = np.array(eos_token_ids, dtype=np.int64)
    prefill_cache_namespace = _prefill_replay_cache_namespace(
        model_dir=gguf_path.parent,
        prompt_length=prompt_length,
        max_sequence_length=max_sequence_length,
    )
    decode_cache_namespace = _decode_replay_cache_namespace(
        model_dir=gguf_path.parent,
        max_sequence_length=max_sequence_length,
    )

    print("Declaring tensors...")
    create_model_tensors(
        prompt_length=prompt_length,
        max_sequence_length=max_sequence_length,
        num_hidden_layers=int(config.num_hidden_layers),
        num_key_value_heads=int(config.num_key_value_heads),
        head_dim=int(config.head_dim),
        max_new_tokens=max_new_tokens,
        eos_token_count=len(eos_token_ids),
        vocab_size=int(config.vocab_size),
    )
    rt = RuntimeSession.open(
        device_index=0,
        model_dir=gguf_path.parent,
        profile_dir=profile_dir,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )
    rt.register_session_tensors({model_tensors().eos_token_ids: eos_token_array})

    zero_cache = np.zeros(
        (
            1,
            int(config.num_key_value_heads),
            max_sequence_length,
            int(config.head_dim),
        ),
        dtype=np.float16,
    )
    rt.materialize_model_weights()

    print(f"Prefill prompt_length={prompt_length}...")
    prefill_inputs = _prefill_inputs(
        input_ids=prepared.input_ids,
        prompt_length=prompt_length,
    )
    with rt.request(
        inputs={tensor.name: value for tensor, value in prefill_inputs.items()},
        state={
            **{
                cache: zero_cache
                for cache in model_tensors().key_caches + model_tensors().value_caches
            },
            model_tensors().generated_tokens: np.zeros((1, max_new_tokens), dtype=np.int64),
            model_tensors().generated_length: np.zeros((1,), dtype=np.uint32),
            model_tensors().stopped: np.zeros((1,), dtype=np.uint32),
        },
    ):
        prefill_replay_plan = _cached_replay_plan(rt, cache_namespace=prefill_cache_namespace)
        prefill_start = time.perf_counter()
        if prefill_replay_plan is None:
            _run_prefill(rt, rope_theta=rope_theta)
            prefill_replay_plan = _build_replay_plan(
                rt,
                name="quantized_qwen3_prefill",
                frame="qwen3.prefill",
                cache_namespace=prefill_cache_namespace,
            )
        else:
            stage_replay_step_inputs(
                rt,
                plan=prefill_replay_plan,
                inputs=prefill_inputs,
                write_through=(model_tensors().text_layers[0].cache_position,),
            )
            execute_replay(
                prefill_replay_plan,
                dynamic_push_constants={"start_position": 0, "token_index": 0},
            )
        first_token = _read_selected_token(rt, model_tensors().next_token)
        prefill_elapsed = time.perf_counter() - prefill_start
        print(f"  first_token={first_token}, prefill={prefill_elapsed:.3f}s")

        decode_replay_plan = _cached_replay_plan(rt, cache_namespace=decode_cache_namespace)
        decode_steps = 0
        decode_start = time.perf_counter()
        for step in range(max_new_tokens - 1):
            cache_pos = prompt_length + step
            if decode_replay_plan is None:
                _run_decode_step(
                    rt,
                    cache_position=cache_pos,
                    rope_theta=rope_theta,
                    step=step,
                )
                decode_replay_plan = _build_replay_plan(
                    rt,
                    name="quantized_qwen3_decode_step",
                    frame=f"qwen3.decode.{step:04d}",
                    cache_namespace=decode_cache_namespace,
                )
            else:
                stage_replay_step_inputs(
                    rt,
                    plan=decode_replay_plan,
                    inputs={},
                )
                execute_replay(
                    decode_replay_plan,
                    dynamic_push_constants={
                        "cache_position": cache_pos,
                        "start_position": cache_pos,
                        "token_index": step + 1,
                    },
                )
            decode_steps += 1

        decode_elapsed = time.perf_counter() - decode_start
        generated_length = int(
            rt.read_request_state(model_tensors().generated_length).reshape(-1)[0]
        )
        generated_tokens = rt.read_request_state(model_tensors().generated_tokens).reshape(-1)[
            :generated_length
        ]
    text = tokenizer.batch_decode(
        np.array([generated_tokens], dtype=np.int64),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    rt.close()

    result = Qwen3RunResult(
        text=text,
        prompt_length=prompt_length,
        generated_tokens=generated_length,
        prefill_elapsed=prefill_elapsed,
        decode_elapsed=decode_elapsed,
    )
    print(
        f"Decode: {decode_steps} steps in {decode_elapsed:.3f}s "
        f"({result.decode_ms_per_token:.1f} ms/token)"
    )
    print(f"Generated text: {text}")
    return result


if __name__ == "__main__":
    output = main()
    print(json.dumps(asdict(output), ensure_ascii=False, indent=2))
