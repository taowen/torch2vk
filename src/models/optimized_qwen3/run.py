"""Standalone Qwen3 Q4_K_M Vulkan generation.

Run from project root:
    uv run python -m models.optimized_qwen3.run
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from transformers import AutoConfig

from models.hf_cache import resolve_cached_model
from models.optimized_qwen3.dispatch.decode_embed import run_decode_embed
from models.optimized_qwen3.dispatch.decode_layer import run_decode_layer
from models.optimized_qwen3.dispatch.decode_norm import run_decode_norm
from models.optimized_qwen3.dispatch.embed_tokens import (
    run_prefill_full_embed,
    run_prefill_tail_embed,
)
from models.optimized_qwen3.dispatch.text_layer import (
    run_prefill_full_mask_opt,
    run_prefill_full_layer,
    run_prefill_tail_mask_opt,
    run_prefill_tail_last_layer_tail,
    run_prefill_tail_layer,
)
from models.optimized_qwen3.dispatch.text_norm import run_text_norm
from models.optimized_qwen3.export_gguf import REPO_ID, export_qwen3_q4_k_m_gguf
from models.optimized_qwen3.input_prep import (
    DEFAULT_PROMPT,
    load_qwen3_tokenizer,
    prepare_qwen3_inputs,
)
from models.optimized_qwen3.shaders.lm_head_q6_k_argmax_partial_f16 import (
    LM_HEAD_Q6_K_ARGMAX_PARTIAL_F16,
)
from models.optimized_qwen3.shaders.llama_matmul_q4_k_f32 import (
    LLAMA_MATMUL_Q4_K_F32_L,
    LLAMA_MATMUL_Q4_K_F32_M,
)
from models.optimized_qwen3.shaders.llama_matmul_q6_k_f32 import (
    LLAMA_MATMUL_Q6_K_F32_L,
    LLAMA_MATMUL_Q6_K_F32_M,
)
from models.optimized_qwen3.shaders.qwen3_token_select_reduce_f32 import (
    QWEN3_TOKEN_SELECT_REDUCE_F32,
)
from models.optimized_qwen3.shaders.qwen3_token_select_reduce_chunks_f32 import (
    QWEN3_TOKEN_SELECT_REDUCE_CHUNKS_F32,
)
from models.optimized_qwen3.shaders.qwen3_token_store_eos_f32 import QWEN3_TOKEN_STORE_EOS_F32
from models.optimized_qwen3.tensors.model import create_model_tensors, model_tensors
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


PREFILL_CHUNK_SIZE = 512
PREFILL_FULL_REPLAY_CACHE = "optimized_qwen3_prefill_full512:v3"
PREFILL_TAIL_REPLAY_CACHE = "optimized_qwen3_prefill_tail:v3"
DECODE_REPLAY_CACHE = "optimized_qwen3_decode_step:v3"
get_shader = make_shader_loader(
    "models.optimized_qwen3.shaders",
    extra_variants=(
        ROPE_TABLE_F32,
        LLAMA_MATMUL_Q4_K_F32_M,
        LLAMA_MATMUL_Q4_K_F32_L,
        LLAMA_MATMUL_Q6_K_F32_M,
        LLAMA_MATMUL_Q6_K_F32_L,
    ),
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


def _run_decode_step(
    rt: RuntimeSession,
    *,
    cache_position: int,
    rope_theta: float,
    step: int,
) -> int:
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
        QWEN3_TOKEN_STORE_EOS_F32(
            rt,
            next_token=tensors.next_token,
            token_index=step + 1,
            done=tensors.done,
            generated_tokens=tensors.generated_tokens,
            generated_length=tensors.generated_length,
            stopped=tensors.stopped,
        )
    return _read_selected_token(rt, tensors.next_token)


def _read_selected_token(rt: RuntimeSession, next_token: LogicalTensor) -> int:
    _require_gpu_output(next_token)
    return int(rt.read_request_state(next_token).reshape(-1)[0])


def _build_decode_replay_plan(
    rt: RuntimeSession,
    *,
    frame: str,
    cache_namespace: str,
) -> ReplayPlan:
    return build_cached_replay_plan(
        rt,
        namespace=cache_namespace,
        name="optimized_qwen3_decode_step",
        frame=frame,
        readback_error="Qwen3 decode replay must not use readback slots",
    )


def _build_prefill_replay_plan(
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
        readback_error="Qwen3 prefill replay must not use readback slots",
    )


def _cached_decode_replay_plan(
    rt: RuntimeSession,
    *,
    cache_namespace: str,
) -> ReplayPlan | None:
    return cached_replay_plan(rt, namespace=cache_namespace)


def _cached_prefill_replay_plan(
    rt: RuntimeSession,
    *,
    cache_namespace: str,
) -> ReplayPlan | None:
    return cached_replay_plan(rt, namespace=cache_namespace)


def _decode_replay_cache_namespace(model_dir: Path) -> str:
    return replay_cache_namespace(
        name=DECODE_REPLAY_CACHE,
        source_digest=_REPLAY_SOURCE_DIGEST,
        model_dir=model_dir,
    )


def _prefill_full_replay_cache_namespace(model_dir: Path) -> str:
    return replay_cache_namespace(
        name=PREFILL_FULL_REPLAY_CACHE,
        source_digest=_REPLAY_SOURCE_DIGEST,
        model_dir=model_dir,
    )


def _prefill_tail_replay_cache_namespace(
    model_dir: Path, tail_length: int, attention_length: int
) -> str:
    return replay_cache_namespace(
        name=PREFILL_TAIL_REPLAY_CACHE,
        source_digest=_REPLAY_SOURCE_DIGEST,
        model_dir=model_dir,
        shape_key=f"tail={tail_length}:attention={attention_length}",
    )


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _prefill_causal_mask(chunk_start: int, chunk_length: int, attention_length: int) -> np.ndarray:
    mask = np.zeros((chunk_length, attention_length), dtype=np.float16)
    for row in range(chunk_length):
        mask[row, chunk_start + row + 1 :] = np.float16(-np.inf)
    return mask


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
    replay_cache_namespace = _decode_replay_cache_namespace(gguf_path.parent)
    config = AutoConfig.from_pretrained(model_dir)
    prompt_length = prepared.prompt_length
    fixed_prefill_length = PREFILL_CHUNK_SIZE if prompt_length > PREFILL_CHUNK_SIZE else 0
    tail_length = prompt_length - fixed_prefill_length
    max_sequence_length = _round_up(prompt_length + 128, 64)
    prefill_attention_length = _round_up(prompt_length, 64)
    prefill_full_cache_namespace = _prefill_full_replay_cache_namespace(gguf_path.parent)
    prefill_tail_cache_namespace = _prefill_tail_replay_cache_namespace(
        gguf_path.parent,
        tail_length,
        prefill_attention_length,
    )
    rope_theta = float(getattr(config, "rope_theta", 1_000_000.0))
    eos_token_ids = (int(config.eos_token_id),)
    eos_token_array = np.array(eos_token_ids, dtype=np.int64)

    print("Declaring tensors...")
    create_model_tensors(
        prompt_length=prompt_length,
        prefill_chunk_length=PREFILL_CHUNK_SIZE,
        prefill_tail_length=tail_length,
        prefill_attention_length=prefill_attention_length,
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

    zero_full_flash_cache = np.zeros(
        (1, PREFILL_CHUNK_SIZE, int(config.num_key_value_heads), int(config.head_dim)),
        dtype=np.float16,
    )
    zero_prefill_flash_cache = np.zeros(
        (1, prefill_attention_length, int(config.num_key_value_heads), int(config.head_dim)),
        dtype=np.float16,
    )
    zero_decode_cache = np.zeros(
        (1, max_sequence_length, int(config.num_key_value_heads), int(config.head_dim)),
        dtype=np.float16,
    )
    rt.initialize_request_state(
        {
            cache: zero_full_flash_cache
            for cache in model_tensors().prefill_full_key_caches
            + model_tensors().prefill_full_value_caches
        }
    )
    rt.initialize_request_state(
        {
            cache: zero_prefill_flash_cache
            for cache in model_tensors().prefill_key_caches + model_tensors().prefill_value_caches
        }
    )
    rt.initialize_request_state(
        {
            cache: zero_decode_cache
            for cache in model_tensors().decode_key_caches + model_tensors().decode_value_caches
        }
    )
    rt.initialize_request_state(
        {
            model_tensors().generated_tokens: np.zeros((1, max_new_tokens), dtype=np.int64),
            model_tensors().generated_length: np.zeros((1,), dtype=np.uint32),
            model_tensors().stopped: np.zeros((1,), dtype=np.uint32),
        }
    )
    rt.materialize_model_weights()

    print(f"Prefill prompt_length={prompt_length}...")
    tail_inputs = {
        model_tensors().prefill_tail_input_ids: prepared.input_ids[:, fixed_prefill_length:],
        model_tensors().prefill_tail_causal_mask: _prefill_causal_mask(
            fixed_prefill_length,
            tail_length,
            prefill_attention_length,
        ),
        model_tensors().prefill_tail_layers[0].cache_position: np.arange(
            fixed_prefill_length,
            prompt_length,
            dtype=np.int64,
        ),
    }
    full_inputs: dict[LogicalTensor, np.ndarray] = {}
    if fixed_prefill_length:
        full_inputs = {
            model_tensors().prefill_full_input_ids: prepared.input_ids[:, :PREFILL_CHUNK_SIZE],
            model_tensors().prefill_full_causal_mask: _prefill_causal_mask(
                0,
                PREFILL_CHUNK_SIZE,
                PREFILL_CHUNK_SIZE,
            ),
            model_tensors().prefill_full_layers[0].cache_position: np.arange(
                PREFILL_CHUNK_SIZE,
                dtype=np.int64,
            ),
        }
    full_prefill_replay_plan = (
        _cached_prefill_replay_plan(rt, cache_namespace=prefill_full_cache_namespace)
        if fixed_prefill_length
        else None
    )
    tail_prefill_replay_plan = _cached_prefill_replay_plan(
        rt,
        cache_namespace=prefill_tail_cache_namespace,
    )
    request_inputs = {tensor.name: value for tensor, value in tail_inputs.items()}
    request_inputs.update({tensor.name: value for tensor, value in full_inputs.items()})
    with rt.request(**request_inputs):
        prefill_start = time.perf_counter()
        if fixed_prefill_length:
            if full_prefill_replay_plan is None:
                with rt.frame("qwen3.prefill.full512"):
                    ROPE_TABLE_F32(
                        rt,
                        start_position=0,
                        theta=rope_theta,
                        cos=model_tensors().prefill_full_rope.cos,
                        sin=model_tensors().prefill_full_rope.sin,
                    )
                    run_prefill_full_embed(rt)
                    run_prefill_full_mask_opt(rt)
                    for layer_idx in range(len(model_tensors().prefill_full_layers)):
                        run_prefill_full_layer(rt, layer_idx)
                _build_prefill_replay_plan(
                    rt,
                    name="optimized_qwen3_prefill_full512",
                    frame="qwen3.prefill.full512",
                    cache_namespace=prefill_full_cache_namespace,
                )
            else:
                stage_replay_step_inputs(
                    rt,
                    plan=full_prefill_replay_plan,
                    inputs=full_inputs,
                    write_through=(model_tensors().prefill_full_layers[0].cache_position,),
                )
                execute_replay(
                    full_prefill_replay_plan,
                    dynamic_push_constants={"start_position": 0},
                )

        if tail_prefill_replay_plan is None:
            with rt.frame("qwen3.prefill.tail"):
                ROPE_TABLE_F32(
                    rt,
                    start_position=fixed_prefill_length,
                    theta=rope_theta,
                    cos=model_tensors().prefill_tail_rope.cos,
                    sin=model_tensors().prefill_tail_rope.sin,
                )
                run_prefill_tail_embed(rt)
                run_prefill_tail_mask_opt(rt)
                for layer_idx in range(len(model_tensors().prefill_tail_layers) - 1):
                    run_prefill_tail_layer(rt, layer_idx)
                run_prefill_tail_last_layer_tail(rt)
                run_text_norm(rt)
                _run_lm_head_select(rt, x=model_tensors().text_norm.mul_1)
                QWEN3_TOKEN_STORE_EOS_F32(
                    rt,
                    next_token=model_tensors().next_token,
                    token_index=0,
                    done=model_tensors().done,
                    generated_tokens=model_tensors().generated_tokens,
                    generated_length=model_tensors().generated_length,
                    stopped=model_tensors().stopped,
                )
            _build_prefill_replay_plan(
                rt,
                name="optimized_qwen3_prefill_tail",
                frame="qwen3.prefill.tail",
                cache_namespace=prefill_tail_cache_namespace,
            )
        else:
            stage_replay_step_inputs(
                rt,
                plan=tail_prefill_replay_plan,
                inputs=tail_inputs,
                write_through=(model_tensors().prefill_tail_layers[0].cache_position,),
            )
            execute_replay(
                tail_prefill_replay_plan,
                dynamic_push_constants={
                    "start_position": fixed_prefill_length,
                    "token_index": 0,
                },
            )
        first_token = _read_selected_token(rt, model_tensors().next_token)
        prefill_elapsed = time.perf_counter() - prefill_start
        print(f"  first_token={first_token}, prefill={prefill_elapsed:.3f}s")

        decode_replay_plan: ReplayPlan | None = _cached_decode_replay_plan(
            rt,
            cache_namespace=replay_cache_namespace,
        )
        decode_steps = 0
        decode_start = time.perf_counter()
        for step in range(max_new_tokens - 1):
            cache_pos = prompt_length + step
            if decode_replay_plan is None:
                decode_replay_plan = _cached_decode_replay_plan(
                    rt,
                    cache_namespace=replay_cache_namespace,
                )
                if decode_replay_plan is None:
                    _run_decode_step(
                        rt,
                        cache_position=cache_pos,
                        rope_theta=rope_theta,
                        step=step,
                    )
                    decode_replay_plan = _build_decode_replay_plan(
                        rt,
                        frame=f"qwen3.decode.{step:04d}",
                        cache_namespace=replay_cache_namespace,
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
