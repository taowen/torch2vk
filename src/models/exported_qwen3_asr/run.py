"""Full ASR pipeline using generated shaders and dispatch functions.

Run from project root:
    .venv/bin/python -m models.exported_qwen3_asr.run
"""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)

from models.hf_cache import resolve_cached_model
from models.optimized_qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.optimized_qwen3_asr.pytorch.example import REPO_ID
from models.optimized_qwen3_asr.shaders.token_select_f32 import QWEN3_ASR_TOKEN_SELECT_GREEDY_F32
from models.optimized_qwen3_asr.shaders.token_store_f32 import QWEN3_ASR_TOKEN_STORE_EOS_F32
from models.exported_qwen3_asr import dispatch
from models.exported_qwen3_asr.tensors import audio_encoder as audio_encoder_tensors
from models.exported_qwen3_asr.tensors import audio_inject as audio_inject_tensors
from models.exported_qwen3_asr.tensors import decode_embed as decode_embed_tensors
from models.exported_qwen3_asr.tensors import decode_layer as decode_layer_tensors
from models.exported_qwen3_asr.tensors import decode_lm_head as decode_lm_head_tensors
from models.exported_qwen3_asr.tensors import decode_norm as decode_norm_tensors
from models.exported_qwen3_asr.tensors import embed_tokens as embed_tokens_tensors
from models.exported_qwen3_asr.tensors import lm_head as lm_head_tensors
from models.exported_qwen3_asr.tensors import rope as rope_tensors
from models.exported_qwen3_asr.tensors import text_layer as text_layer_tensors
from models.exported_qwen3_asr.tensors import text_norm as text_norm_tensors
from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    TensorSemantic,
    TensorSpec,
)
from torch2vk.runtime.replay import ReplayPlan, execute_replay, stage_replay_step_inputs
from torch2vk.runtime.session import RuntimeSession


def _require_gpu_output(tensor: LogicalTensor) -> None:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} did not produce a GPU buffer")


def _host_input_tensor(name: str, dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )


def _request_output_tensor(name: str, dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.OUTPUT,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )


def _request_state_tensor(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    *,
    semantic: TensorSemantic | None = None,
) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
        semantic=semantic,
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
    processor, prepared = prepare_qwen3_asr_inputs(model_dir=model_dir, wav=str(wav_path))
    prompt_length = prepared.prompt_length
    max_sequence_length = prepared.prompt_length + 64
    rt = RuntimeSession.open(device_index=0, model_dir=model_dir)

    # === Create all tensor objects upfront ===
    print("Declaring tensors...")

    # Audio encoder (conv + layers + proj as single dispatch)
    audio_encoder_t = audio_encoder_tensors.create_audio_encoder(
        "spike.audio",
        request_state_outputs={audio_encoder_tensors.AUDIO_ENCODER_OUTPUT},
    )

    # Text (non-layered)
    embed_tokens_t = embed_tokens_tensors.create_embed_tokens("spike.text.embed")
    audio_inject_t = audio_inject_tensors.create_audio_inject(
        "spike.text.audio_inject",
        audio_features=audio_encoder_t.linear_110,
        index_copy=embed_tokens_t.embedding,
    )
    key_caches = tuple(
        _request_state_tensor(
            f"spike.text.layers.{layer_idx}.key_cache",
            "float32",
            (1, tc.num_key_value_heads, max_sequence_length, tc.head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for layer_idx in range(tc.num_hidden_layers)
    )
    value_caches = tuple(
        _request_state_tensor(
            f"spike.text.layers.{layer_idx}.value_cache",
            "float32",
            (1, tc.num_key_value_heads, max_sequence_length, tc.head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for layer_idx in range(tc.num_hidden_layers)
    )
    prefill_rope_t = rope_tensors.create_rope_table(
        "spike.text.prefill.rope",
        batch=1,
        sequence_length=prompt_length,
        head_dim=tc.head_dim,
    )
    decode_rope_t = rope_tensors.create_rope_table(
        "spike.decode.rope",
        batch=1,
        sequence_length=1,
        head_dim=tc.head_dim,
    )

    # Text layers (layered)
    text_layer_ts = []
    text_hidden = audio_inject_t.index_copy
    for layer_idx in range(tc.num_hidden_layers):
        layer_tensors = text_layer_tensors.create_text_layer(
            f"spike.text.layer.{layer_idx}",
            layer_idx=layer_idx,
            hidden_states=text_hidden,
            index_copy=key_caches[layer_idx],
            index_copy_1=value_caches[layer_idx],
            position_embeddings_0=prefill_rope_t.cos,
            position_embeddings_1=prefill_rope_t.sin,
            cache_position=text_layer_ts[0].cache_position if layer_idx > 0 else None,
        )
        text_layer_ts.append(layer_tensors)
        text_hidden = layer_tensors.add_7
    text_norm_t = text_norm_tensors.create_text_norm(
        "spike.text.norm",
        hidden_states=text_layer_ts[-1].add_7,
    )
    lm_head_t = lm_head_tensors.create_lm_head(
        "spike.text.lm_head",
        input=text_norm_t.mul_1,
        request_state_outputs={lm_head_tensors.LM_HEAD_OUTPUT},
    )

    # Decode (non-layered, reuse for all decode steps)
    decode_embed_t = decode_embed_tensors.create_decode_embed(
        "spike.decode.embed",
        p_weight=embed_tokens_t.p_weight,
    )

    # Decode layers (layered)
    decode_layer_ts = []
    decode_hidden = decode_embed_t.embedding
    for layer_idx in range(tc.num_hidden_layers):
        prefill_layer_tensors = text_layer_ts[layer_idx]
        layer_tensors = decode_layer_tensors.create_decode_layer(
            f"spike.decode.layer.{layer_idx}",
            layer_idx=layer_idx,
            p_input_layernorm_weight=prefill_layer_tensors.p_input_layernorm_weight,
            p_post_attention_layernorm_weight=(
                prefill_layer_tensors.p_post_attention_layernorm_weight
            ),
            p_attn_q_proj_weight=prefill_layer_tensors.p_attn_q_proj_weight,
            p_attn_k_proj_weight=prefill_layer_tensors.p_attn_k_proj_weight,
            p_attn_v_proj_weight=prefill_layer_tensors.p_attn_v_proj_weight,
            p_attn_o_proj_weight=prefill_layer_tensors.p_attn_o_proj_weight,
            p_attn_q_norm_weight=prefill_layer_tensors.p_attn_q_norm_weight,
            p_attn_k_norm_weight=prefill_layer_tensors.p_attn_k_norm_weight,
            p_mlp_gate_proj_weight=prefill_layer_tensors.p_mlp_gate_proj_weight,
            p_mlp_up_proj_weight=prefill_layer_tensors.p_mlp_up_proj_weight,
            p_mlp_down_proj_weight=prefill_layer_tensors.p_mlp_down_proj_weight,
            hidden_states=decode_hidden,
            index_copy=key_caches[layer_idx],
            index_copy_1=value_caches[layer_idx],
            position_embeddings_0=decode_rope_t.cos,
            position_embeddings_1=decode_rope_t.sin,
            cache_position=decode_layer_ts[0].cache_position if layer_idx > 0 else None,
        )
        decode_layer_ts.append(layer_tensors)
        decode_hidden = layer_tensors.add_7
    decode_norm_t = decode_norm_tensors.create_decode_norm(
        "spike.decode.norm",
        p_weight=text_norm_t.p_weight,
        hidden_states=decode_layer_ts[-1].add_7,
    )
    decode_lm_head_t = decode_lm_head_tensors.create_decode_lm_head(
        "spike.decode.lm_head",
        p_weight=lm_head_t.p_weight,
        input=decode_norm_t.mul_1,
        request_state_outputs={decode_lm_head_tensors.DECODE_LM_HEAD_OUTPUT},
    )

    zero_cache = np.zeros(
        (1, tc.num_key_value_heads, max_sequence_length, tc.head_dim),
        dtype=np.float32,
    )
    rt.initialize_request_state(
        {cache: zero_cache for cache in key_caches + value_caches}
    )

    # === Audio Tower ===
    print("\n=== Phase 1: Audio Tower ===")
    feat_len = prepared.audio_feature_length
    input_features = np.ascontiguousarray(prepared.input_features[0, :, :feat_len], dtype=np.float32)
    audio_feature_lens = np.array([feat_len], dtype=np.int64)

    n_window = 50
    chunk_num = int(np.ceil(feat_len / (n_window * 2)))
    chunk_lengths = np.full(chunk_num, n_window * 2, dtype=np.int64)
    remainder = feat_len % (n_window * 2)
    if remainder != 0:
        chunk_lengths[-1] = remainder

    features_t = input_features.T
    chunks = []
    offset = 0
    for cl in chunk_lengths:
        chunks.append(features_t[offset:offset + cl])
        offset += cl

    max_chunk_len = int(chunk_lengths.max())
    num_mel = input_features.shape[0]
    padded_feature = np.zeros((chunk_num, 1, num_mel, max_chunk_len), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        padded_feature[i, 0, :, :chunk.shape[0]] = chunk.T

    # Positional embedding + compact
    pos_shape = tuple(int(dim) for dim in audio_encoder_t.position_embedding.spec.shape)
    _, t_dim, _ = pos_shape
    pos_emb = _compute_positional_embedding(t_dim, ac.d_model)
    position_embedding = np.ascontiguousarray(
        np.broadcast_to(pos_emb[None, :t_dim, :], pos_shape),
        dtype=np.float32,
    )

    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    valid_positions = []
    for i, fl in enumerate(feature_lens_after_cnn):
        for j in range(int(fl)):
            valid_positions.append(i * t_dim + j)
    compact_index = np.array(valid_positions, dtype=np.int64)
    print(f"  hidden_states after compact: {audio_encoder_t.index_select.spec.shape}")

    # Build block-diagonal attention mask from cu_seqlens
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
    attention_mask = np.full((1, 1, seq_len, seq_len), -np.finfo(np.float32).max, dtype=np.float32)
    for i in range(1, len(cu_seqlens)):
        s, e = int(cu_seqlens[i - 1]), int(cu_seqlens[i])
        attention_mask[0, 0, s:e, s:e] = 0.0

    # Run entire audio encoder in one frame
    print(f"  audio encoder ({padded_feature.shape})...")
    rt._inputs.clear()
    rt.register_inputs(
        {
            audio_encoder_t.x: padded_feature,
            audio_encoder_t.position_embedding: position_embedding,
            audio_encoder_t.compact_index: compact_index,
            audio_encoder_t.attention_mask: attention_mask,
        }
    )
    if pytorch_compare:
        audio_frame_scope = rt.frame(
            "spike.audio",
            pytorch_model_class=Qwen3ASRForConditionalGeneration,
            pytorch_model_submodule="thinker.audio_tower",
            pytorch_args=(input_features, audio_feature_lens),
        )
    else:
        audio_frame_scope = rt.frame("spike.audio")
    with audio_frame_scope:
        dispatch.run_audio_encoder(rt, audio_encoder_t)
    audio_hidden_t = audio_encoder_t.linear_110
    _require_gpu_output(audio_hidden_t)
    print(f"  Audio tower output: {audio_hidden_t.spec.shape}")

    # === Text Prefill ===
    print("\n=== Phase 2: Text Prefill ===")
    prompt_length = prepared.prompt_length

    ids_flat = prepared.input_ids.flatten()
    audio_positions = np.where(ids_flat == 151676)[0].astype(np.int64)
    if len(audio_positions) > 0:
        audio_start = int(audio_positions[0])
        audio_end = audio_start + len(audio_positions)
        print(f"    Injecting audio [{audio_start}:{audio_end}]")

    rt.register_inputs(
        {
            prefill_rope_t.start_position: np.array([0], dtype=np.int64),
            prefill_rope_t.theta: np.array([5_000_000.0], dtype=np.float32),
        }
    )
    dispatch.run_rope_table(
        rt,
        prefill_rope_t,
        frame_name="spike.text.prefill_rope",
    )

    # Embedding, audio injection, and decoder layers stay on GPU.
    print(f"  embed + audio inject + decoder layers x {tc.num_hidden_layers}...")
    rt._inputs.clear()
    rt.register_inputs(
        {
            embed_tokens_t.input: prepared.input_ids.astype(np.int64),
            audio_inject_t.audio_positions: audio_positions,
            text_layer_ts[0].cache_position: np.arange(prompt_length, dtype=np.int64),
        }
    )
    if pytorch_compare:
        prefill_frame_scope = rt.frame(
            "spike.text.prefill",
            pytorch_model_class=Qwen3ASRForConditionalGeneration,
            pytorch_model_submodule="thinker",
            pytorch_kwargs={
                "input_ids": prepared.input_ids,
                "attention_mask": prepared.attention_mask,
                "input_features": prepared.input_features,
                "feature_attention_mask": prepared.feature_attention_mask,
                "use_cache": True,
            },
            pytorch_cache_policy="hf_dynamic",
            pytorch_cache_namespace="exported_qwen3_asr.text",
            pytorch_reset_cache=True,
        )
    else:
        prefill_frame_scope = rt.frame("spike.text.prefill")
    with prefill_frame_scope:
        dispatch.run_embed_tokens(rt, embed_tokens_t)
        dispatch.run_audio_inject(rt, audio_inject_t)
        for layer_idx, layer_tensors in enumerate(text_layer_ts):
            dispatch.run_text_layer(rt, layer_tensors)
            if layer_idx % 7 == 6:
                print(f"    layer {layer_idx} done")
        dispatch.run_text_norm(rt, text_norm_t)
        dispatch.run_lm_head(rt, lm_head_t)


    print("  lm_head + token_select...")
    logits_t = lm_head_t.linear
    _require_gpu_output(logits_t)

    eos_token_ids = (151645, 151643)
    eos_token_array = np.array(eos_token_ids, dtype=np.int64)
    eos_token_ids_t = _host_input_tensor(
        "spike.token_select.eos_token_ids",
        "int64",
        (len(eos_token_ids),),
    )
    next_token_t = _request_output_tensor("spike.token_select.next_token", "int64", (1,))
    done_t = _request_output_tensor("spike.token_select.done", "uint32", (1,))
    generated_tokens_t = _request_state_tensor(
        "spike.token_select.generated_tokens",
        "int64",
        (1, max_new_tokens),
        semantic=TensorSemantic.TOKEN,
    )
    generated_length_t = _request_state_tensor(
        "spike.token_select.generated_length",
        "uint32",
        (1,),
        semantic=TensorSemantic.TOKEN,
    )
    stopped_t = _request_state_tensor(
        "spike.token_select.stopped",
        "uint32",
        (1,),
        semantic=TensorSemantic.TOKEN,
    )
    token_index_t = _host_input_tensor("spike.token_select.token_index", "int64", (1,))
    rt.initialize_request_state(
        {
            generated_tokens_t: np.zeros((1, max_new_tokens), dtype=np.int64),
            generated_length_t: np.zeros((1,), dtype=np.uint32),
            stopped_t: np.zeros((1,), dtype=np.uint32),
        }
    )
    rt.register_inputs({eos_token_ids_t: eos_token_array})
    _run_token_select(
        rt,
        logits=logits_t,
        eos_token_ids=eos_token_ids_t,
        next_token=next_token_t,
        done=done_t,
        frame_name="spike.text.token_select",
    )
    rt.register_inputs({token_index_t: np.array([0], dtype=np.int64)})
    _run_token_store(
        rt,
        next_token=next_token_t,
        token_index=token_index_t,
        done=done_t,
        generated_tokens=generated_tokens_t,
        generated_length=generated_length_t,
        stopped=stopped_t,
        frame_name="spike.text.token_store",
    )
    first_token = _read_selected_token(rt, next_token_t)
    print(f"  First token: {first_token}")

    # === Decode Loop ===
    print("\n=== Phase 3: Decode Loop ===")
    eos_token_set = set(eos_token_ids)
    generated_tokens = [first_token]
    decode_replay_plan: ReplayPlan | None = None
    decode_tensors_by_name = _collect_tensors_by_name(
        decode_embed_t,
        decode_layer_ts,
        decode_norm_t,
        decode_lm_head_t,
        eos_token_ids_t,
        next_token_t,
        done_t,
        generated_tokens_t,
        generated_length_t,
        stopped_t,
        token_index_t,
    )

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
                decode_rope_t.start_position: np.array([cache_pos], dtype=np.int64),
                decode_rope_t.theta: np.array([5_000_000.0], dtype=np.float32),
            }
        )
        dispatch.run_rope_table(
            rt,
            decode_rope_t,
            frame_name=f"spike.decode.rope.{step:04d}",
        )

        decode_step_inputs = dispatch.decode_step_inputs(
            decode_embed_t=decode_embed_t,
            decode_layer_ts=decode_layer_ts,
            eos_token_ids=eos_token_ids_t,
            token_index=token_index_t,
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
                decode_embed_t=decode_embed_t,
                decode_layer_ts=decode_layer_ts,
                decode_norm_t=decode_norm_t,
                decode_lm_head_t=decode_lm_head_t,
                eos_token_ids=eos_token_ids_t,
                next_token=next_token_t,
                done=done_t,
                token_index=token_index_t,
                generated_tokens=generated_tokens_t,
                generated_length=generated_length_t,
                stopped=stopped_t,
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
                    token_feedback_source=next_token_t,
                    token_feedback_target=decode_embed_t.input,
                )
        else:
            stage_replay_step_inputs(
                rt,
                plan=decode_replay_plan,
                tensors_by_name=decode_tensors_by_name,
                inputs=decode_step_inputs,
                write_through=(decode_layer_ts[0].cache_position, token_index_t),
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
    stored_length = int(rt.read_request_state(generated_length_t).reshape(-1)[0])
    generated_tokens = [
        int(token)
        for token in rt.read_request_state(generated_tokens_t).reshape(-1)[:stored_length]
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
