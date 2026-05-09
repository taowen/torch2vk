"""Full ASR pipeline using generated shaders and dispatch functions.

Run from project root:
    .venv/bin/python -m models.exported_qwen3_asr.run
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import cast
import numpy as np
import torch
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput

from models.hf_cache import resolve_cached_model
from models.optimized_qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.optimized_qwen3_asr.pytorch.example import REPO_ID
from models.optimized_qwen3_asr.shaders.token_select_f32 import QWEN3_ASR_TOKEN_SELECT_GREEDY_F32
from models.optimized_qwen3_asr.shaders.token_store_f32 import QWEN3_ASR_TOKEN_STORE_EOS_F32
from models.exported_qwen3_asr import dispatch
from models.exported_qwen3_asr.export_forwards import (
    export_audio_tower_forward,
    patched_forward,
)
from models.exported_qwen3_asr.tensors.model import create_model_tensors, model_tensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.replay import ReplayPlan, execute_replay, stage_replay_step_inputs
from torch2vk.runtime.session import RuntimeSession


def _require_gpu_output(tensor: LogicalTensor) -> None:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} did not produce a GPU buffer")


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
        QWEN3_ASR_TOKEN_SELECT_GREEDY_F32(
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
        QWEN3_ASR_TOKEN_STORE_EOS_F32(
            rt,
            next_token=next_token,
            token_index=token_index,
            done=done,
            generated_tokens=generated_tokens,
            generated_length=generated_length,
            stopped=stopped,
        )


def _read_selected_token(rt: RuntimeSession, next_token: LogicalTensor) -> int:
    _require_gpu_output(next_token)
    return int(rt.read_request_state(next_token).reshape(-1)[0])


# ==============================================================
# Audio tower helpers (CPU ops)
# ==============================================================

def _get_feat_extract_output_lengths(input_lengths: np.ndarray) -> np.ndarray:
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def _compute_positional_embedding(length: int, channels: int) -> np.ndarray:
    max_timescale = 10000.0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = np.exp(-log_timescale_increment * np.arange(channels // 2, dtype=np.float32))
    scaled_time = np.arange(length, dtype=np.float32)[:, None] * inv_timescales[None, :]
    return np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1).astype(np.float32)


def _preprocess_audio_inputs(
    input_ids: np.ndarray,
    input_features: np.ndarray,
    feature_attention_mask: np.ndarray,
    *,
    position_embedding_shape: tuple[int, ...],
    d_model: int,
) -> dict[str, np.ndarray]:
    feat_len = int(np.asarray(feature_attention_mask).sum(axis=-1).reshape(-1)[0])
    mel = np.ascontiguousarray(np.asarray(input_features)[0, :, :feat_len], dtype=np.float32)

    n_window = 50
    chunk_num = int(np.ceil(feat_len / (n_window * 2)))
    chunk_lengths = np.full(chunk_num, n_window * 2, dtype=np.int64)
    remainder = feat_len % (n_window * 2)
    if remainder != 0:
        chunk_lengths[-1] = remainder

    features_t = mel.T
    chunks = []
    offset = 0
    for chunk_length in chunk_lengths:
        chunk = features_t[offset : offset + chunk_length]
        chunks.append(chunk)
        offset += int(chunk_length)

    max_chunk_len = int(chunk_lengths.max())
    num_mel = mel.shape[0]
    padded_feature = np.zeros((chunk_num, 1, num_mel, max_chunk_len), dtype=np.float32)
    for index, chunk in enumerate(chunks):
        padded_feature[index, 0, :, : chunk.shape[0]] = chunk.T

    _, t_dim, _ = position_embedding_shape
    pos_emb = _compute_positional_embedding(t_dim, d_model)
    position_embedding = np.ascontiguousarray(
        np.broadcast_to(pos_emb[None, :t_dim, :], position_embedding_shape),
        dtype=np.float32,
    )

    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    valid_positions = [
        chunk_index * t_dim + offset
        for chunk_index, feature_len in enumerate(feature_lens_after_cnn)
        for offset in range(int(feature_len))
    ]
    compact_index = np.array(valid_positions, dtype=np.int64)

    n_window_infer = 800
    aftercnn_lens = _get_feat_extract_output_lengths(np.array([feat_len], dtype=np.int64))
    window_aftercnn = int(feature_lens_after_cnn.max()) * (n_window_infer // (n_window * 2))
    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens:
        cnn_len = int(cnn_len)
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        rem = cnn_len % window_aftercnn
        if rem != 0:
            cu_chunk_lens += [rem]
    cu_seqlens = np.cumsum(cu_chunk_lens, dtype=np.int32)

    seq_len = compact_index.shape[0]
    attention_mask = np.full(
        (1, 1, seq_len, seq_len),
        -np.finfo(np.float32).max,
        dtype=np.float32,
    )
    for index in range(1, len(cu_seqlens)):
        start, end = int(cu_seqlens[index - 1]), int(cu_seqlens[index])
        attention_mask[0, 0, start:end, start:end] = 0.0

    audio_positions = np.where(np.asarray(input_ids).reshape(-1) == 151676)[0].astype(np.int64)
    return {
        "padded_feature": padded_feature,
        "position_embedding": position_embedding,
        "compact_index": compact_index,
        "audio_attention_mask": attention_mask,
        "cu_seqlens": cu_seqlens,
        "audio_positions": audio_positions,
    }


def _audio_position_embedding_shape(
    *,
    feature_length: int,
    d_model: int,
) -> tuple[int, int, int]:
    n_window = 50
    chunk_num = int(np.ceil(feature_length / (n_window * 2)))
    chunk_lengths = np.full(chunk_num, n_window * 2, dtype=np.int64)
    remainder = feature_length % (n_window * 2)
    if remainder != 0:
        chunk_lengths[-1] = remainder
    time = int(_get_feat_extract_output_lengths(chunk_lengths).max())
    return (chunk_num, time, d_model)


def _debug_audio_tower_forward(
    self: torch.nn.Module,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor | None = None,
    aftercnn_lens: torch.Tensor | None = None,
) -> BaseModelOutput:
    if feature_lens is None:
        raise ValueError("feature_lens is required for exported_qwen3_asr audio debug")
    config = getattr(self, "config")
    d_model = int(getattr(config, "d_model"))
    feat_len = int(feature_lens.detach().cpu().reshape(-1)[0].item())
    features = input_features.detach().cpu().float().numpy()
    arrays = _preprocess_audio_inputs(
        np.zeros((1, 0), dtype=np.int64),
        features[None, :, :feat_len],
        np.ones((1, feat_len), dtype=np.int64),
        position_embedding_shape=_audio_position_embedding_shape(
            feature_length=feat_len,
            d_model=d_model,
        ),
        d_model=d_model,
    )
    device = input_features.device
    dtype = input_features.dtype
    output = export_audio_tower_forward(
        self,
        torch.from_numpy(arrays["padded_feature"]).to(device=device, dtype=dtype),
        torch.from_numpy(arrays["position_embedding"]).to(device=device, dtype=dtype),
        torch.from_numpy(arrays["compact_index"]).to(device=device),
        torch.from_numpy(arrays["cu_seqlens"]).to(device=device),
        torch.from_numpy(arrays["audio_attention_mask"]).to(device=device, dtype=dtype),
    )
    return BaseModelOutput(
        last_hidden_state=cast(torch.FloatTensor, output["last_hidden_state"])
    )



def _collect_tensors_by_name(*values: object) -> dict[str, LogicalTensor]:
    tensors_by_name: dict[str, LogicalTensor] = {}

    def collect(value: object) -> None:
        if isinstance(value, LogicalTensor):
            tensors_by_name[value.name] = value
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                collect(getattr(value, field.name))
            return
        if isinstance(value, Iterable) and not isinstance(value, str | bytes):
            for item in value:
                collect(item)

    for value in values:
        collect(value)
    return tensors_by_name


def _build_decode_replay_plan(
    rt: RuntimeSession,
    *,
    dispatch_start: int,
    dispatch_end: int,
    tensors_by_name: dict[str, LogicalTensor],
    token_feedback_source: LogicalTensor,
    token_feedback_target: LogicalTensor,
) -> ReplayPlan:
    warmup_records = rt.dispatch_records[dispatch_start:dispatch_end]
    variants = [dispatch.shader_variant(record.shader) for record in warmup_records]
    plan = rt.build_replay_plan(
        name="exported_qwen3_asr_decode_step",
        frame_dispatch_records=list(warmup_records),
        variants=variants,
        tensors_by_name=tensors_by_name,
        token_feedback_source=token_feedback_source,
        token_feedback_target=token_feedback_target,
    )
    if plan.readback_slots:
        plan.close()
        raise RuntimeError("Spike Qwen3-ASR decode replay must not use readback slots")
    rt.cache_replay_plan("exported_qwen3_asr_decode_step:v1", plan)
    return plan


# ==============================================================
# Main pipeline
# ==============================================================

def main(
    *,
    use_replay: bool = True,
    pytorch_compare: bool = False,
    max_new_tokens: int = 64,
) -> str:
    if max_new_tokens <= 0 or max_new_tokens > 64:
        raise ValueError(f"max_new_tokens must be in [1, 64], got {max_new_tokens}")
    if pytorch_compare:
        use_replay = False
    wav_path = Path("tests/fixtures/qwen3_asr_asknot.wav")
    if not wav_path.exists():
        raise FileNotFoundError(f"Test wav not found at {wav_path}")

    print("Preparing inputs...")
    model_dir = resolve_cached_model(REPO_ID)
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
    processor, prepared = prepare_qwen3_asr_inputs(model_dir=model_dir, wav=str(wav_path))
    prompt_length = prepared.prompt_length
    max_sequence_length = prepared.prompt_length + 64
    rt = RuntimeSession.open(device_index=0, model_dir=model_dir)
    pytorch_audio_tower = None
    pytorch_thinker = None
    if pytorch_compare:
        pytorch_model = rt._load_pytorch_model(Qwen3ASRForConditionalGeneration)
        if pytorch_model is None:
            raise RuntimeError("exported_qwen3_asr compare requires a PyTorch model")
        pytorch_audio_tower = getattr(getattr(pytorch_model, "thinker"), "audio_tower")
        pytorch_thinker = getattr(pytorch_model, "thinker")

    # === Create all tensor objects upfront ===
    print("Declaring tensors...")
    eos_token_ids = (151645, 151643)
    create_model_tensors(
        input_ids_shape=tuple(int(dim) for dim in prepared.input_ids.shape),
        attention_mask_shape=tuple(int(dim) for dim in prepared.attention_mask.shape),
        input_features_shape=tuple(int(dim) for dim in prepared.input_features.shape),
        feature_attention_mask_shape=tuple(
            int(dim) for dim in prepared.feature_attention_mask.shape
        ),
        prompt_length=prompt_length,
        max_sequence_length=max_sequence_length,
        num_hidden_layers=tc.num_hidden_layers,
        num_key_value_heads=tc.num_key_value_heads,
        head_dim=tc.head_dim,
        max_new_tokens=max_new_tokens,
        eos_token_count=len(eos_token_ids),
    )

    zero_cache = np.zeros(
        (1, tc.num_key_value_heads, max_sequence_length, tc.head_dim),
        dtype=np.float32,
    )
    rt.initialize_request_state(
        {cache: zero_cache for cache in model_tensors().key_caches + model_tensors().value_caches}
    )

    # === Audio Tower ===
    print("\n=== Phase 1: Audio Tower ===")
    preprocessed = _preprocess_audio_inputs(
        prepared.input_ids,
        prepared.input_features,
        prepared.feature_attention_mask,
        position_embedding_shape=tuple(
            int(dim) for dim in model_tensors().audio_encoder_t.position_embedding.spec.shape
        ),
        d_model=ac.d_model,
    )
    print(f"  hidden_states after compact: {model_tensors().audio_encoder_t.index_select.spec.shape}")
    print(f"  audio encoder ({model_tensors().audio_encoder_t.x.spec.shape})...")
    rt._inputs.clear()
    rt.register_inputs(
        {
            model_tensors().audio_encoder_t.x: preprocessed["padded_feature"],
            model_tensors().audio_encoder_t.position_embedding: preprocessed["position_embedding"],
            model_tensors().audio_encoder_t.compact_index: preprocessed["compact_index"],
            model_tensors().audio_encoder_t.attention_mask: preprocessed["audio_attention_mask"],
        }
    )
    if pytorch_audio_tower is not None:
        with patched_forward(pytorch_audio_tower, export_audio_tower_forward):
            with rt.frame("spike.audio", pytorch_model=pytorch_audio_tower):
                dispatch.run_audio_encoder(rt)
    else:
        with rt.frame("spike.audio"):
            dispatch.run_audio_encoder(rt)
    _require_gpu_output(model_tensors().audio_encoder_t.linear_110)
    print(f"  Audio tower output: {model_tensors().audio_encoder_t.linear_110.spec.shape}")

    # === Text Prefill ===
    print("\n=== Phase 2: Text Prefill ===")
    prompt_length = prepared.prompt_length

    audio_positions = preprocessed["audio_positions"]
    if len(audio_positions) > 0:
        audio_start = int(audio_positions[0])
        audio_end = audio_start + len(audio_positions)
        print(f"    Injecting audio [{audio_start}:{audio_end}]")

    rt._inputs.clear()
    rt.register_inputs(
        {
            model_tensors().prefill_rope_t.start_position: np.array([0], dtype=np.int64),
            model_tensors().prefill_rope_t.theta: np.array([rope_theta], dtype=np.float32),
        }
    )
    dispatch.run_rope_table(
        rt,
        phase="prefill",
        frame_name="spike.text.prefill_rope",
    )

    # Embedding, audio injection, and decoder layers stay on GPU.
    print(f"  embed + audio inject + decoder layers x {tc.num_hidden_layers}...")
    rt._inputs.clear()
    if pytorch_compare:
        prefill_frame_scope = rt.frame(
            "spike.text.prefill",
            pytorch_model=pytorch_thinker,
            pytorch_cache_policy="hf_dynamic",
            pytorch_cache_namespace="exported_qwen3_asr.text",
            pytorch_reset_cache=True,
        )
    else:
        prefill_frame_scope = rt.frame("spike.text.prefill")
    prefill_audio_patch_scope = (
        patched_forward(pytorch_audio_tower, _debug_audio_tower_forward)
        if pytorch_audio_tower is not None
        else nullcontext()
    )
    with prefill_audio_patch_scope:
        with prefill_frame_scope:
            rt.register_inputs(
                {
                    model_tensors().input_ids_t: np.ascontiguousarray(
                        prepared.input_ids,
                        dtype=np.int64,
                    ),
                    model_tensors().attention_mask_t: np.ascontiguousarray(
                        prepared.attention_mask,
                        dtype=np.int64,
                    ),
                    model_tensors().input_features_t: np.ascontiguousarray(
                        prepared.input_features,
                        dtype=np.float32,
                    ),
                    model_tensors().feature_attention_mask_t: np.ascontiguousarray(
                        prepared.feature_attention_mask,
                        dtype=np.int64,
                    ),
                    model_tensors().position_ids_t: np.broadcast_to(
                        np.arange(prompt_length, dtype=np.int64)[None, None, :],
                        (3, 1, prompt_length),
                    ).copy(),
                    model_tensors().audio_inject_t.audio_positions: preprocessed["audio_positions"],
                    model_tensors().text_layer_ts[0].cache_position: np.arange(
                        prompt_length,
                        dtype=np.int64,
                    ),
                }
            )
            dispatch.run_embed_tokens(rt)
            dispatch.run_audio_inject(rt)
            for layer_idx in range(len(model_tensors().text_layer_ts)):
                dispatch.run_text_layer(rt, layer_idx)
                if layer_idx % 7 == 6:
                    print(f"    layer {layer_idx} done")
            dispatch.run_text_norm(rt)
            dispatch.run_lm_head(rt)

    print("  lm_head + token_select...")
    _require_gpu_output(model_tensors().lm_head_t.linear)

    eos_token_array = np.array(eos_token_ids, dtype=np.int64)
    rt.initialize_request_state(
        {
            model_tensors().generated_tokens_t: np.zeros((1, max_new_tokens), dtype=np.int64),
            model_tensors().generated_length_t: np.zeros((1,), dtype=np.uint32),
            model_tensors().stopped_t: np.zeros((1,), dtype=np.uint32),
        }
    )
    rt.register_inputs({model_tensors().eos_token_ids_t: eos_token_array})
    _run_token_select(
        rt,
        logits=model_tensors().lm_head_t.linear,
        eos_token_ids=model_tensors().eos_token_ids_t,
        next_token=model_tensors().next_token_t,
        done=model_tensors().done_t,
        frame_name="spike.text.token_select",
    )
    rt.register_inputs({model_tensors().token_index_t: np.array([0], dtype=np.int64)})
    _run_token_store(
        rt,
        next_token=model_tensors().next_token_t,
        token_index=model_tensors().token_index_t,
        done=model_tensors().done_t,
        generated_tokens=model_tensors().generated_tokens_t,
        generated_length=model_tensors().generated_length_t,
        stopped=model_tensors().stopped_t,
        frame_name="spike.text.token_store",
    )
    first_token = _read_selected_token(rt, model_tensors().next_token_t)
    print(f"  First token: {first_token}")

    # === Decode Loop ===
    print("\n=== Phase 3: Decode Loop ===")
    eos_token_set = set(eos_token_ids)
    generated_tokens = [first_token]
    decode_replay_plan: ReplayPlan | None = None
    decode_tensors_by_name = _collect_tensors_by_name(model_tensors())

    # Memory sampling
    memory_trace: list[tuple[int, float, float, float]] = []
    baseline_stats = rt.device.allocation_stats()
    print(f"  Baseline GPU memory: device_local={baseline_stats.device_local_live_bytes / 1024**2:.1f} MB, "
          f"reserved={baseline_stats.device_local_reserved_bytes / 1024**2:.1f} MB")

    decode_start = time.perf_counter()
    for step in range(max_new_tokens - 1):
        if generated_tokens[-1] in eos_token_set:
            print(f"  EOS at step {step}")
            break

        cache_pos = prompt_length + step

        rt.register_inputs(
            {
                model_tensors().decode_rope_t.start_position: np.array(
                    [cache_pos],
                    dtype=np.int64,
                ),
                model_tensors().decode_rope_t.theta: np.array(
                    [rope_theta],
                    dtype=np.float32,
                ),
            }
        )
        dispatch.run_rope_table(
            rt,
            phase="decode",
            frame_name=f"spike.decode.rope.{step:04d}",
        )

        decode_step_inputs = dispatch.decode_step_inputs(
            token=generated_tokens[-1],
            cache_position=cache_pos,
            eos_token_array=eos_token_array,
            token_index_value=step + 1,
        )
        if decode_replay_plan is None:
            dispatch_start = len(rt.dispatch_records)
            rt.register_inputs(decode_step_inputs)
            next_token = dispatch.run_decode_step(
                rt,
                step=step,
            )
            generated_tokens.append(next_token)
            dispatch_end = len(rt.dispatch_records)
            if use_replay:
                decode_replay_plan = _build_decode_replay_plan(
                    rt,
                    dispatch_start=dispatch_start,
                    dispatch_end=dispatch_end,
                    tensors_by_name=decode_tensors_by_name,
                    token_feedback_source=model_tensors().next_token_t,
                    token_feedback_target=model_tensors().decode_embed_t.input,
                )
        else:
            stage_replay_step_inputs(
                rt,
                plan=decode_replay_plan,
                tensors_by_name=decode_tensors_by_name,
                inputs=decode_step_inputs,
                write_through=(
                    model_tensors().decode_layer_ts[0].cache_position,
                    model_tensors().token_index_t,
                ),
            )
            execute_replay(decode_replay_plan)

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
            if use_replay:
                print(f"  Step {step}: gpu={stats.device_local_live_bytes / 1024**2:.1f}MB")
            else:
                print(f"  Step {step}: token={generated_tokens[-1]}  "
                      f"gpu={stats.device_local_live_bytes / 1024**2:.1f}MB")

    decode_elapsed = time.perf_counter() - decode_start
    decode_steps = len(memory_trace)
    print(f"\n  Decode: {decode_steps} steps in {decode_elapsed:.3f}s "
          f"({decode_elapsed / decode_steps * 1000:.1f} ms/token)" if decode_steps > 0 else "")

    # Memory summary
    print("\n=== GPU Memory Trace ===")
    if memory_trace:
        final_stats = rt.device.allocation_stats()
        print(f"  Peak device_local live: {final_stats.device_local_peak_live_bytes / 1024**2:.1f} MB")
        print(f"  Peak device_local reserved: {final_stats.device_local_peak_reserved_bytes / 1024**2:.1f} MB")
        print(f"  Final device_local live: {final_stats.device_local_live_bytes / 1024**2:.1f} MB")
        print(f"  Steps sampled: {len(memory_trace)}")
        print()
        print("  Step | GPU Live (MB) | GPU Reserved (MB) | Host Upload (MB)")
        print("  -----|---------------|-------------------|------------------")
        for step, device_local_live_mb, device_local_reserved_mb, host_upload_live_mb in memory_trace:
            print(f"  {step:4d} | {device_local_live_mb:13.1f} | "
                  f"{device_local_reserved_mb:17.1f} | {host_upload_live_mb:.1f}")

    # Decode text
    print("\n=== Result ===")
    stored_length = int(rt.read_request_state(model_tensors().generated_length_t).reshape(-1)[0])
    generated_tokens = [
        int(token)
        for token in rt.read_request_state(model_tensors().generated_tokens_t).reshape(-1)[
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
