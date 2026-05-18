"""Full ASR pipeline using generated shaders and dispatch functions.

Run from project root:
    .venv/bin/python -m models.quantized_qwen3_asr.run
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig

from models.hf_cache import resolve_cached_model
from models.optimized_qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.optimized_qwen3_asr.pytorch.example import REPO_ID
from models.quantized_qwen3_asr.export_gguf import export_qwen3_asr_q4_k_m_gguf
from models.quantized_qwen3_asr.dispatch.audio_encoder import run_audio_encoder
from models.quantized_qwen3_asr.dispatch.audio_inject import run_audio_inject
from models.quantized_qwen3_asr.dispatch.decode_embed import run_decode_embed
from models.quantized_qwen3_asr.dispatch.decode_layer import run_decode_layer
from models.quantized_qwen3_asr.dispatch.decode_norm import run_decode_norm
from models.quantized_qwen3_asr.dispatch.embed_tokens import run_embed_tokens
from models.quantized_qwen3_asr.dispatch.text_layer import run_text_layer
from models.quantized_qwen3_asr.dispatch.text_norm import run_text_norm
from models.quantized_qwen3_asr.pytorch_modules import (
    audio_position_embedding_shape,
    preprocess_audio_inputs,
)
from models.quantized_qwen3_asr.shaders.lm_head_q6_k_argmax_partial_f16 import (
    LM_HEAD_Q6_K_ARGMAX_PARTIAL_F16,
)
from models.quantized_qwen3_asr.shaders.qwen3_asr_token_store_eos import (
    QWEN3_ASR_TOKEN_STORE_EOS,
)
from models.quantized_qwen3_asr.shaders.qwen3_token_select_reduce_chunks_f32 import (
    QWEN3_TOKEN_SELECT_REDUCE_CHUNKS_F32,
)
from models.quantized_qwen3_asr.shaders.qwen3_token_select_reduce_f32 import (
    QWEN3_TOKEN_SELECT_REDUCE_F32,
)
from models.quantized_qwen3_asr.shaders.slice_last_token_f16 import SLICE_LAST_TOKEN_F16
from models.quantized_qwen3_asr.tensors.model import create_model_tensors, model_tensors
from torch2vk.runtime.host_array import as_float16_array, as_float16_attention_mask
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.replay import ReplayPlan, execute_replay, stage_replay_step_inputs
from torch2vk.runtime.replay_cache_key import (
    build_cached_replay_plan,
    cached_replay_plan,
    replay_cache_namespace,
    source_tree_digest,
)
from torch2vk.runtime.rope_table import run_rope_table_f32
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader_loader import make_shader_loader

DECODE_REPLAY_CACHE = "quantized_qwen3_asr_decode_step:v6"
_STOP_CHECK_INTERVAL = 2
get_shader = make_shader_loader("models.quantized_qwen3_asr.shaders")


_REPLAY_SOURCE_DIGEST = source_tree_digest(__file__)


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


def _slice_prefill_lm_head_input(rt: RuntimeSession) -> None:
    tensors = model_tensors()
    SLICE_LAST_TOKEN_F16(
        rt,
        x=tensors.text_norm.rms_norm,
        output=tensors.prefill_lm_head_input,
    )


def _run_token_store(
    rt: RuntimeSession,
    *,
    next_token: LogicalTensor,
    token_index: int,
    done: LogicalTensor,
    generated_tokens: LogicalTensor,
    generated_length: LogicalTensor,
    stopped: LogicalTensor,
    frame_name: str,
) -> None:
    with rt.frame(frame_name):
        QWEN3_ASR_TOKEN_STORE_EOS(
            rt,
            next_token=next_token,
            token_index=token_index,
            done=done,
            generated_tokens=generated_tokens,
            generated_length=generated_length,
            stopped=stopped,
        )


def _run_decode_step(rt: RuntimeSession, *, cache_position: int, step: int) -> int:
    tensors = model_tensors()
    if not tensors.decode_layers:
        raise ValueError("decode_layers must not be empty")
    with rt.frame(f"spike.decode.{step:04d}"):
        run_decode_embed(rt)
        for layer_idx in range(len(tensors.decode_layers)):
            run_decode_layer(rt, layer_idx, cache_position=cache_position)
        run_decode_norm(rt)
        _run_lm_head_select(rt, x=tensors.decode_norm.rms_norm)
        QWEN3_ASR_TOKEN_STORE_EOS(
            rt,
            next_token=tensors.next_token,
            token_index=step + 1,
            done=tensors.done,
            generated_tokens=tensors.generated_tokens,
            generated_length=tensors.generated_length,
            stopped=tensors.stopped,
        )
    return _read_selected_token(rt, tensors.next_token)


def _run_rope_table(
    rt: RuntimeSession,
    *,
    phase: str,
    start_position: int,
    rope_theta: float,
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
        start_position=start_position,
        theta=rope_theta,
        cos=rope_t.cos,
        sin=rope_t.sin,
        frame_name=frame_name,
    )


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
    return build_cached_replay_plan(
        rt,
        namespace=cache_namespace,
        name="quantized_qwen3_asr_decode_step",
        frame=frame,
        readback_error="Spike Qwen3-ASR decode replay must not use readback slots",
    )


def _cached_decode_replay_plan(
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


# ==============================================================
# Main pipeline
# ==============================================================


def main(
    *,
    max_new_tokens: int = 64,
    wav_path: str | Path = Path("tests/fixtures/qwen3_asr_asknot.wav"),
    profile_dir: str | Path | None = None,
) -> str:
    if max_new_tokens <= 0 or max_new_tokens > 64:
        raise ValueError(f"max_new_tokens must be in [1, 64], got {max_new_tokens}")
    resolved_wav_path = Path(wav_path)
    if not resolved_wav_path.exists():
        raise FileNotFoundError(f"Test wav not found at {resolved_wav_path}")

    print("Preparing inputs...")
    model_dir = resolve_cached_model(REPO_ID)
    gguf_path = export_qwen3_asr_q4_k_m_gguf(model_dir=model_dir)
    replay_cache_namespace = _decode_replay_cache_namespace(gguf_path.parent)
    config_payload = (Path(model_dir) / "config.json").read_text()

    devnull = open(os.devnull, "w")
    stdout_fd, stderr_fd = os.dup(1), os.dup(2)
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    try:
        config = Qwen3ASRConfig(**json.loads(config_payload))
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        devnull.close()

    ac = config.thinker_config.audio_config
    tc = config.thinker_config.text_config
    rope_theta = float(getattr(tc, "rope_theta", 5_000_000.0))
    processor, prepared = prepare_qwen3_asr_inputs(model_dir=model_dir, wav=str(resolved_wav_path))
    prompt_length = prepared.prompt_length
    max_sequence_length = prepared.prompt_length + 64
    audio_feature_length = int(
        np.asarray(prepared.feature_attention_mask).sum(axis=-1).reshape(-1)[0]
    )
    audio_position_shape = audio_position_embedding_shape(
        feature_length=audio_feature_length,
        d_model=ac.d_model,
    )
    preprocessed = preprocess_audio_inputs(
        prepared.input_ids,
        prepared.input_features,
        prepared.feature_attention_mask,
        position_embedding_shape=audio_position_shape,
        d_model=ac.d_model,
    )
    audio_chunk_count = int(preprocessed["padded_feature"].shape[0])
    audio_sequence_length = int(preprocessed["compact_index"].shape[0])

    # === Create all tensor objects upfront ===
    print("Declaring tensors...")
    eos_token_ids = (151645, 151643)
    eos_token_array = np.array(eos_token_ids, dtype=np.int64)
    create_model_tensors(
        input_ids_shape=tuple(int(dim) for dim in prepared.input_ids.shape),
        attention_mask_shape=tuple(int(dim) for dim in prepared.attention_mask.shape),
        input_features_shape=tuple(int(dim) for dim in prepared.input_features.shape),
        feature_attention_mask_shape=tuple(
            int(dim) for dim in prepared.feature_attention_mask.shape
        ),
        prompt_length=prompt_length,
        audio_chunk_count=audio_chunk_count,
        audio_sequence_length=audio_sequence_length,
        max_sequence_length=max_sequence_length,
        num_hidden_layers=tc.num_hidden_layers,
        num_key_value_heads=tc.num_key_value_heads,
        head_dim=tc.head_dim,
        max_new_tokens=max_new_tokens,
        eos_token_count=len(eos_token_ids),
        vocab_size=int(tc.vocab_size),
    )
    rt = RuntimeSession.open(
        device_index=0,
        model_dir=gguf_path.parent,
        profile_dir=profile_dir,
        model_tensors=model_tensors(),
        session_tensors={model_tensors().eos_token_ids: eos_token_array},
        get_shader=get_shader,
    )
    zero_cache = np.zeros(
        (1, tc.num_key_value_heads, max_sequence_length, tc.head_dim),
        dtype=np.float16,
    )
    prefill_position_ids = np.broadcast_to(
        np.arange(prompt_length, dtype=np.int64)[None, None, :],
        (3, 1, prompt_length),
    ).copy()
    prefill_cache_position = np.arange(prompt_length, dtype=np.int64)
    with rt.request(
        inputs={
            "input_ids": np.ascontiguousarray(prepared.input_ids, dtype=np.int64),
            "attention_mask": np.ascontiguousarray(prepared.attention_mask, dtype=np.int64),
            "input_features": np.ascontiguousarray(prepared.input_features, dtype=np.float32),
            "feature_attention_mask": np.ascontiguousarray(
                prepared.feature_attention_mask,
                dtype=np.int64,
            ),
            model_tensors().audio_encoder.x.name: as_float16_array(preprocessed["padded_feature"]),
            model_tensors().audio_encoder.position_embedding.name: as_float16_array(
                preprocessed["position_embedding"]
            ),
            model_tensors().audio_encoder.compact_index.name: preprocessed["compact_index"],
            model_tensors().audio_encoder.attention_mask.name: as_float16_attention_mask(
                preprocessed["audio_attention_mask"]
            ),
            model_tensors().position_ids.name: prefill_position_ids,
            model_tensors().audio_inject.audio_positions.name: preprocessed["audio_positions"],
            model_tensors().text_layers[0].cache_position.name: prefill_cache_position,
        },
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

        # === Audio Tower ===
        print("\n=== Phase 1: Audio Tower ===")
        print(f"  hidden_states after compact: {model_tensors().audio_encoder.index_select.spec.shape}")
        print(f"  audio encoder ({model_tensors().audio_encoder.x.spec.shape})...")
        with rt.frame("spike.audio"):
            run_audio_encoder(rt)
        _require_gpu_output(model_tensors().audio_encoder.linear_110)
        print(f"  Audio tower output: {model_tensors().audio_encoder.linear_110.spec.shape}")

        # === Text Prefill ===
        print("\n=== Phase 2: Text Prefill ===")
        prompt_length = prepared.prompt_length

        audio_positions = preprocessed["audio_positions"]
        if len(audio_positions) > 0:
            audio_start = int(audio_positions[0])
            audio_end = audio_start + len(audio_positions)
            print(f"    Injecting audio [{audio_start}:{audio_end}]")

        _run_rope_table(
            rt,
            phase="prefill",
            start_position=0,
            rope_theta=rope_theta,
            frame_name="spike.text.prefill_rope",
        )

        # Embedding, audio injection, and decoder layers stay on GPU.
        print(f"  embed + audio inject + decoder layers x {tc.num_hidden_layers}...")
        with rt.frame("spike.text.prefill"):
            run_embed_tokens(rt)
            run_audio_inject(rt)
            for layer_idx in range(len(model_tensors().text_layers)):
                run_text_layer(rt, layer_idx)
                if layer_idx % 7 == 6:
                    print(f"    layer {layer_idx} done")
            run_text_norm(rt)
            _slice_prefill_lm_head_input(rt)
            _run_lm_head_select(rt, x=model_tensors().prefill_lm_head_input)

        print("  lm_head + token_select...")
        _run_token_store(
            rt,
            next_token=model_tensors().next_token,
            token_index=0,
            done=model_tensors().done,
            generated_tokens=model_tensors().generated_tokens,
            generated_length=model_tensors().generated_length,
            stopped=model_tensors().stopped,
            frame_name="spike.text.token_store",
        )
        first_token = _read_selected_token(rt, model_tensors().next_token)
        print(f"  First token: {first_token}")

        # === Decode Loop ===
        print("\n=== Phase 3: Decode Loop ===")
        decode_replay_plan: ReplayPlan | None = None

        # Memory sampling
        memory_trace: list[tuple[int, float, float, float]] = []
        baseline_stats = rt.device.allocation_stats()
        print(
            f"  Baseline GPU memory: device_local={baseline_stats.device_local_live_bytes / 1024**2:.1f} MB, "
            f"reserved={baseline_stats.device_local_reserved_bytes / 1024**2:.1f} MB"
        )

        decode_start = time.perf_counter()
        for step in range(max_new_tokens - 1):
            if _request_stopped(rt):
                print(f"  EOS at step {step}")
                break

            cache_pos = prompt_length + step

            _run_rope_table(
                rt,
                phase="decode",
                start_position=cache_pos,
                rope_theta=rope_theta,
                frame_name=f"spike.decode.rope.{step:04d}",
            )

            if decode_replay_plan is None:
                decode_replay_plan = _cached_decode_replay_plan(
                    rt,
                    cache_namespace=replay_cache_namespace,
                )
                if decode_replay_plan is None:
                    _run_decode_step(
                        rt,
                        cache_position=cache_pos,
                        step=step,
                    )
                    decode_replay_plan = _build_decode_replay_plan(
                        rt,
                        frame=f"spike.decode.{step:04d}",
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
                        "token_index": step + 1,
                    },
                )

            stats = rt.device.allocation_stats()
            memory_trace.append(
                (
                    step,
                    stats.device_local_live_bytes / 1024**2,
                    stats.device_local_reserved_bytes / 1024**2,
                    stats.host_upload_live_bytes / 1024**2,
                )
            )

            if step < 5 or step % 20 == 0:
                print(f"  Step {step}: gpu={stats.device_local_live_bytes / 1024**2:.1f}MB")

            if (step + 1) % _STOP_CHECK_INTERVAL == 0 and _request_stopped(rt):
                print(f"  EOS at step {step + 1}")
                break

        decode_elapsed = time.perf_counter() - decode_start
        decode_steps = len(memory_trace)
        print(
            f"\n  Decode: {decode_steps} steps in {decode_elapsed:.3f}s "
            f"({decode_elapsed / decode_steps * 1000:.1f} ms/token)"
            if decode_steps > 0
            else ""
        )

        # Memory summary
        print("\n=== GPU Memory Trace ===")
        if memory_trace:
            final_stats = rt.device.allocation_stats()
            print(
                f"  Peak device_local live: {final_stats.device_local_peak_live_bytes / 1024**2:.1f} MB"
            )
            print(
                f"  Peak device_local reserved: {final_stats.device_local_peak_reserved_bytes / 1024**2:.1f} MB"
            )
            print(f"  Final device_local live: {final_stats.device_local_live_bytes / 1024**2:.1f} MB")
            print(f"  Steps sampled: {len(memory_trace)}")
            print()
            print("  Step | GPU Live (MB) | GPU Reserved (MB) | Host Upload (MB)")
            print("  -----|---------------|-------------------|------------------")
            for (
                step,
                device_local_live_mb,
                device_local_reserved_mb,
                host_upload_live_mb,
            ) in memory_trace:
                print(
                    f"  {step:4d} | {device_local_live_mb:13.1f} | "
                    f"{device_local_reserved_mb:17.1f} | {host_upload_live_mb:.1f}"
                )

        # Decode text
        print("\n=== Result ===")
        stored_length = int(rt.read_request_state(model_tensors().generated_length).reshape(-1)[0])
        generated_tokens = [
            int(token)
            for token in rt.read_request_state(model_tensors().generated_tokens).reshape(-1)[
                :stored_length
            ]
        ]
        print(f"Generated {len(generated_tokens)} tokens")
        text = processor.batch_decode(
            np.array([generated_tokens], dtype=np.int64),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print(f"Transcription: {text}")

    rt.close()
    return text


if __name__ == "__main__":
    result = main()
    print(result)
