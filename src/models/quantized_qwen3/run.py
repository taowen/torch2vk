"""Standalone Qwen3 Q4_K_M Vulkan generation.

Run from project root:
    uv run python -m models.quantized_qwen3.run
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from transformers import AutoConfig

from models.hf_cache import resolve_cached_model
from models.quantized_qwen3.dispatch.decode_embed import run_decode_embed
from models.quantized_qwen3.dispatch.decode_layer import run_decode_layer
from models.quantized_qwen3.dispatch.decode_lm_head import run_decode_lm_head
from models.quantized_qwen3.dispatch.decode_norm import run_decode_norm
from models.quantized_qwen3.dispatch.embed_tokens import run_embed_tokens
from models.quantized_qwen3.dispatch.lm_head import run_lm_head
from models.quantized_qwen3.dispatch.text_layer import run_text_layer
from models.quantized_qwen3.dispatch.text_norm import run_text_norm
from models.quantized_qwen3.export_gguf import REPO_ID, export_qwen3_q4_k_m_gguf
from models.quantized_qwen3.input_prep import (
    DEFAULT_PROMPT,
    load_qwen3_tokenizer,
    prepare_qwen3_inputs,
)
from models.quantized_qwen3.shaders.qwen3_token_select_greedy_f32 import (
    QWEN3_TOKEN_SELECT_GREEDY_F32,
)
from models.quantized_qwen3.shaders.qwen3_token_store_eos_f32 import QWEN3_TOKEN_STORE_EOS_F32
from models.quantized_qwen3.shaders.slice_last_token_f16 import SLICE_LAST_TOKEN_F16
from models.quantized_qwen3.tensors.model import create_model_tensors, model_tensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.replay import ReplayPlan, execute_replay, stage_replay_step_inputs
from torch2vk.runtime.rope_table import ROPE_TABLE_F32, run_rope_table_f32
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader import ShaderVariant
from torch2vk.runtime.shader_loader import make_shader_loader


PREFILL_REPLAY_CACHE = "quantized_qwen3_prefill:v3"
DECODE_REPLAY_CACHE = "quantized_qwen3_decode_step:v2"
_STOP_CHECK_INTERVAL = 16
_load_model_shader = make_shader_loader("models.quantized_qwen3.shaders")


def get_shader(name: str) -> ShaderVariant:
    if name == ROPE_TABLE_F32.name:
        return ROPE_TABLE_F32
    return _load_model_shader(name)


def _source_tree_digest() -> str:
    root = Path(__file__).parent
    hasher = hashlib.sha256()
    for path in (
        Path(__file__),
        *sorted((root / "dispatch").glob("*.py")),
        *sorted((root / "shaders").glob("*.py")),
        *sorted((root / "tensors").glob("*.py")),
    ):
        hasher.update(str(path.relative_to(root)).encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


_REPLAY_SOURCE_DIGEST = _source_tree_digest()


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


def _run_rope_table(
    rt: RuntimeSession,
    *,
    phase: str,
    frame_name: str,
) -> None:
    tensors = model_tensors()
    if phase == "prefill":
        rope_t = tensors.prefill_rope
    elif phase == "decode":
        rope_t = tensors.decode_rope
    else:
        raise ValueError(f"unknown rope phase: {phase}")
    run_rope_table_f32(
        rt,
        start_position=rope_t.start_position,
        theta=rope_t.theta,
        cos=rope_t.cos,
        sin=rope_t.sin,
        frame_name=frame_name,
    )


def _run_token_select(
    rt: RuntimeSession,
    *,
    logits: LogicalTensor,
    eos_token_ids: LogicalTensor,
    next_token: LogicalTensor,
    done: LogicalTensor,
    frame_name: str,
) -> LogicalTensor:
    with rt.frame(frame_name):
        QWEN3_TOKEN_SELECT_GREEDY_F32(
            rt,
            logits=logits,
            eos_token_ids=eos_token_ids,
            next_token=next_token,
            done=done,
        )
    return next_token


def _run_token_store(
    rt: RuntimeSession,
    *,
    next_token: LogicalTensor,
    token_index: LogicalTensor,
    done: LogicalTensor,
    generated_tokens: LogicalTensor,
    generated_length: LogicalTensor,
    stopped: LogicalTensor,
    frame_name: str,
) -> None:
    with rt.frame(frame_name):
        QWEN3_TOKEN_STORE_EOS_F32(
            rt,
            next_token=next_token,
            token_index=token_index,
            done=done,
            generated_tokens=generated_tokens,
            generated_length=generated_length,
            stopped=stopped,
        )


def _run_decode_step(rt: RuntimeSession, *, step: int) -> int:
    tensors = model_tensors()
    with rt.frame(f"qwen3.decode.{step:04d}"):
        ROPE_TABLE_F32(
            rt,
            start_position=tensors.decode_rope.start_position,
            theta=tensors.decode_rope.theta,
            cos=tensors.decode_rope.cos,
            sin=tensors.decode_rope.sin,
        )
        run_decode_embed(rt)
        for layer_idx in range(len(tensors.decode_layers)):
            run_decode_layer(rt, layer_idx)
        run_decode_norm(rt)
        run_decode_lm_head(rt)
        QWEN3_TOKEN_SELECT_GREEDY_F32(
            rt,
            logits=tensors.decode_lm_head.linear,
            eos_token_ids=tensors.eos_token_ids,
            next_token=tensors.next_token,
            done=tensors.done,
        )
        QWEN3_TOKEN_STORE_EOS_F32(
            rt,
            next_token=tensors.next_token,
            token_index=tensors.token_index,
            done=tensors.done,
            generated_tokens=tensors.generated_tokens,
            generated_length=tensors.generated_length,
            stopped=tensors.stopped,
        )
    return _read_selected_token(rt, tensors.next_token)


def _decode_step_inputs(
    *,
    cache_position: int,
    rope_theta: float,
    eos_token_array: np.ndarray,
    token_index_value: int,
) -> dict[LogicalTensor, np.ndarray]:
    tensors = model_tensors()
    return {
        tensors.decode_rope.start_position: np.array([cache_position], dtype=np.int64),
        tensors.decode_rope.theta: np.array([rope_theta], dtype=np.float32),
        tensors.decode_layers[0].cache_position: np.array([cache_position], dtype=np.int64),
        tensors.eos_token_ids: np.ascontiguousarray(eos_token_array, dtype=np.int64),
        tensors.token_index: np.array([token_index_value], dtype=np.int64),
    }


def _read_selected_token(rt: RuntimeSession, next_token: LogicalTensor) -> int:
    _require_gpu_output(next_token)
    return int(rt.read_request_state(next_token).reshape(-1)[0])


def _request_stopped(rt: RuntimeSession) -> bool:
    return bool(rt.read_request_state(model_tensors().stopped).reshape(-1)[0])


def _build_decode_replay_plan(
    rt: RuntimeSession,
    *,
    frame: str,
    cache_namespace: str,
) -> ReplayPlan:
    plan = rt.build_replay_plan(
        name="quantized_qwen3_decode_step",
        frame=frame,
    )
    if plan.readback_slots:
        plan.close()
        raise RuntimeError("Qwen3 decode replay must not use readback slots")
    rt.cache_replay_plan(cache_namespace, plan)
    return plan


def _build_prefill_replay_plan(
    rt: RuntimeSession,
    *,
    frame: str,
    cache_namespace: str,
) -> ReplayPlan:
    plan = rt.build_replay_plan(
        name="quantized_qwen3_prefill",
        frame=frame,
    )
    if plan.readback_slots:
        plan.close()
        raise RuntimeError("Qwen3 prefill replay must not use readback slots")
    rt.cache_replay_plan(cache_namespace, plan)
    return plan


def _cached_decode_replay_plan(
    rt: RuntimeSession,
    *,
    cache_namespace: str,
) -> ReplayPlan | None:
    for plan in rt.cached_replay_plans(cache_namespace):
        return plan
    return None


def _cached_prefill_replay_plan(
    rt: RuntimeSession,
    *,
    cache_namespace: str,
) -> ReplayPlan | None:
    for plan in rt.cached_replay_plans(cache_namespace):
        return plan
    return None


def _decode_replay_cache_namespace(model_dir: Path) -> str:
    return f"{DECODE_REPLAY_CACHE}:{_REPLAY_SOURCE_DIGEST}:{model_dir.resolve()}"


def _prefill_replay_cache_namespace(model_dir: Path, prompt_length: int) -> str:
    return f"{PREFILL_REPLAY_CACHE}:{_REPLAY_SOURCE_DIGEST}:{model_dir.resolve()}:prompt={prompt_length}"


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
    prefill_cache_namespace = _prefill_replay_cache_namespace(
        gguf_path.parent,
        prepared.prompt_length,
    )
    replay_cache_namespace = _decode_replay_cache_namespace(gguf_path.parent)
    config = AutoConfig.from_pretrained(model_dir)
    prompt_length = prepared.prompt_length
    max_sequence_length = prompt_length + 128
    rope_theta = float(getattr(config, "rope_theta", 1_000_000.0))
    eos_token_ids = (int(config.eos_token_id),)

    print("Declaring tensors...")
    create_model_tensors(
        prompt_length=prompt_length,
        max_sequence_length=max_sequence_length,
        num_hidden_layers=int(config.num_hidden_layers),
        num_key_value_heads=int(config.num_key_value_heads),
        head_dim=int(config.head_dim),
        max_new_tokens=max_new_tokens,
        eos_token_count=len(eos_token_ids),
    )
    rt = RuntimeSession.open(
        device_index=0,
        model_dir=gguf_path.parent,
        profile_dir=profile_dir,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )

    zero_cache = np.zeros(
        (1, int(config.num_key_value_heads), max_sequence_length, int(config.head_dim)),
        dtype=np.float16,
    )
    rt.initialize_request_state(
        {cache: zero_cache for cache in model_tensors().key_caches + model_tensors().value_caches}
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
    eos_token_array = np.array(eos_token_ids, dtype=np.int64)
    prefill_cache_position = np.arange(prompt_length, dtype=np.int64)
    prefill_inputs = {
        model_tensors().input_ids: prepared.input_ids,
        model_tensors().text_layers[0].cache_position: prefill_cache_position,
    }
    rt.register_inputs(prefill_inputs)
    prefill_replay_plan = _cached_prefill_replay_plan(
        rt,
        cache_namespace=prefill_cache_namespace,
    )
    rt.register_inputs(
        {
            model_tensors().prefill_rope.start_position: np.array([0], dtype=np.int64),
            model_tensors().prefill_rope.theta: np.array([rope_theta], dtype=np.float32),
        }
    )
    prefill_start = time.perf_counter()
    _run_rope_table(rt, phase="prefill", frame_name="qwen3.prefill.rope")
    if prefill_replay_plan is None:
        with rt.frame("qwen3.prefill"):
            run_embed_tokens(rt)
            for layer_idx in range(len(model_tensors().text_layers)):
                run_text_layer(rt, layer_idx)
            run_text_norm(rt)
            SLICE_LAST_TOKEN_F16(
                rt,
                x=model_tensors().text_norm.mul_1,
                output=model_tensors().prefill_lm_head_input,
            )
            run_lm_head(rt)
        _build_prefill_replay_plan(
            rt,
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
        execute_replay(prefill_replay_plan)
    _require_gpu_output(model_tensors().lm_head.linear)
    rt.register_inputs({model_tensors().eos_token_ids: eos_token_array})
    _run_token_select(
        rt,
        logits=model_tensors().lm_head.linear,
        eos_token_ids=model_tensors().eos_token_ids,
        next_token=model_tensors().next_token,
        done=model_tensors().done,
        frame_name="qwen3.prefill.token_select",
    )
    rt.register_inputs({model_tensors().token_index: np.array([0], dtype=np.int64)})
    _run_token_store(
        rt,
        next_token=model_tensors().next_token,
        token_index=model_tensors().token_index,
        done=model_tensors().done,
        generated_tokens=model_tensors().generated_tokens,
        generated_length=model_tensors().generated_length,
        stopped=model_tensors().stopped,
        frame_name="qwen3.prefill.token_store",
    )
    first_token = _read_selected_token(rt, model_tensors().next_token)
    prefill_elapsed = time.perf_counter() - prefill_start
    print(f"  first_token={first_token}, prefill={prefill_elapsed:.3f}s")

    first_decode_inputs = _decode_step_inputs(
        cache_position=prompt_length,
        rope_theta=rope_theta,
        eos_token_array=eos_token_array,
        token_index_value=1,
    )
    rt.register_inputs(first_decode_inputs)
    decode_replay_plan: ReplayPlan | None = _cached_decode_replay_plan(
        rt,
        cache_namespace=replay_cache_namespace,
    )
    decode_steps = 0
    decode_start = time.perf_counter()
    should_decode = not _request_stopped(rt)
    for step in range(max_new_tokens - 1):
        if not should_decode:
            break
        cache_pos = prompt_length + step
        decode_step_inputs = _decode_step_inputs(
            cache_position=cache_pos,
            rope_theta=rope_theta,
            eos_token_array=eos_token_array,
            token_index_value=step + 1,
        )
        if decode_replay_plan is None:
            rt.register_inputs(decode_step_inputs)
            decode_replay_plan = _cached_decode_replay_plan(
                rt,
                cache_namespace=replay_cache_namespace,
            )
            if decode_replay_plan is None:
                _run_decode_step(rt, step=step)
                decode_replay_plan = _build_decode_replay_plan(
                    rt,
                    frame=f"qwen3.decode.{step:04d}",
                    cache_namespace=replay_cache_namespace,
                )
            else:
                stage_replay_step_inputs(
                    rt,
                    plan=decode_replay_plan,
                    inputs=decode_step_inputs,
                    write_through=(
                        model_tensors().decode_layers[0].cache_position,
                        model_tensors().token_index,
                    ),
                )
                execute_replay(decode_replay_plan)
        else:
            stage_replay_step_inputs(
                rt,
                plan=decode_replay_plan,
                inputs=decode_step_inputs,
                write_through=(
                    model_tensors().decode_layers[0].cache_position,
                    model_tensors().token_index,
                ),
            )
            execute_replay(decode_replay_plan)
        decode_steps += 1
        should_decode = (step + 1) % _STOP_CHECK_INTERVAL != 0 or not _request_stopped(rt)

    decode_elapsed = time.perf_counter() - decode_start
    generated_length = int(rt.read_request_state(model_tensors().generated_length).reshape(-1)[0])
    generated_tokens = rt.read_request_state(model_tensors().generated_tokens).reshape(-1)[:generated_length]
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
