"""Full ASR pipeline using generated shaders and dispatch functions.

Run from project root:
    .venv/bin/python -m models.spike_qwen3_asr.run
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import torch
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
from safetensors import safe_open

from models.hf_cache import resolve_cached_model
from models.qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.qwen3_asr.pytorch.example import REPO_ID
from models.qwen3_asr.shaders.token_select_f32 import QWEN3_ASR_TOKEN_SELECT_GREEDY_F32
from models.spike_qwen3_asr import dispatch
from models.spike_qwen3_asr.tensors import audio_tower as at_tensors
from models.spike_qwen3_asr.tensors import decode as decode_tensors
from models.spike_qwen3_asr.tensors import decode_layer as decode_layer_tensors
from models.spike_qwen3_asr.tensors import encoder_layer as enc_tensors
from models.spike_qwen3_asr.tensors import text as text_tensors
from models.spike_qwen3_asr.tensors import text_layer as text_layer_tensors
from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    TensorSemantic,
    TensorSpec,
)
from torch2vk.runtime.session import RuntimeSession


def _materialize_weights(rt: RuntimeSession, tensors_obj, weights) -> None:
    """Upload float32-converted weights to persistent GPU memory (once per tensor)."""
    for f in dataclasses.fields(tensors_obj):
        tensor: LogicalTensor = getattr(tensors_obj, f.name)
        if tensor.role is not TensorRole.WEIGHT:
            continue
        if tensor.buffer is not None:
            continue
        data = weights.get_tensor(tensor.name).float().numpy()
        ((slice_, alloc),) = rt.device.upload_numpy_arrays_with_allocations(
            [(tensor.name, data)]
        )
        with tensor.runtime_write_scope():
            tensor.buffer = slice_
            tensor.descriptor_nbytes = slice_.nbytes
        rt._model_allocations.append(alloc)


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
) -> int:
    with rt.frame(frame_name):
        QWEN3_ASR_TOKEN_SELECT_GREEDY_F32(
            rt,
            logits=logits,
            eos_token_ids=eos_token_ids,
            next_token=next_token,
            done=done,
        )
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




def _compute_rope(seq_len: int, head_dim: int, rope_theta: float = 5_000_000.0, start_pos: int = 0) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = torch.cos(emb).reshape(1, seq_len, head_dim).numpy()
    sin = torch.sin(emb).reshape(1, seq_len, head_dim).numpy()
    return cos, sin


# ==============================================================
# Main pipeline
# ==============================================================

def main() -> str:
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
    max_new_tokens = 64

    processor, prepared = prepare_qwen3_asr_inputs(model_dir=model_dir, wav=str(wav_path))
    max_sequence_length = prepared.prompt_length + max_new_tokens
    weights = safe_open(str(Path(model_dir) / "model.safetensors"), framework="pt", device="cpu")
    rt = RuntimeSession.open(device_index=0, model_dir=model_dir)

    # === Create all tensor objects upfront and materialize weights ===
    print("Materializing weights...")

    # Audio tower (non-layered)
    conv2d1_t = at_tensors.create_conv2d1("spike.audio.conv2d1")
    conv2d2_t = at_tensors.create_conv2d2(
        "spike.audio.conv2d2",
        bindings={"x": conv2d1_t.gelu},
    )
    conv2d3_t = at_tensors.create_conv2d3(
        "spike.audio.conv2d3",
        bindings={"x": conv2d2_t.gelu},
    )
    conv_out_t = at_tensors.create_conv_out(
        "spike.audio.conv_out",
        bindings={"x": conv2d3_t.gelu},
    )
    audio_position_compact_t = at_tensors.create_audio_position_compact(
        "spike.audio.position_compact",
        bindings={"x": conv_out_t.linear},
        request_state_outputs={at_tensors.AUDIO_POSITION_COMPACT_OUTPUT},
    )

    # Encoder layers (layered)
    encoder_layer_ts = []
    encoder_hidden = audio_position_compact_t.index_select
    encoder_attention_mask = None
    for layer_idx in range(ac.encoder_layers):
        bindings = {"hidden_states": encoder_hidden}
        if encoder_attention_mask is not None:
            bindings["attention_mask"] = encoder_attention_mask
        layer_tensors = enc_tensors.create_encoder_layer(
            f"spike.audio.enc.{layer_idx}",
            layer_idx=layer_idx,
            bindings=bindings,
        )
        encoder_layer_ts.append(layer_tensors)
        if encoder_attention_mask is None:
            encoder_attention_mask = layer_tensors.attention_mask
        encoder_hidden = layer_tensors.add_1
    ln_post_t = at_tensors.create_ln_post(
        "spike.audio.ln_post",
        bindings={"input": encoder_layer_ts[-1].add_1},
    )
    proj1_t = at_tensors.create_proj1(
        "spike.audio.proj1",
        bindings={"x": ln_post_t.layer_norm},
    )
    proj2_t = at_tensors.create_proj2(
        "spike.audio.proj2",
        bindings={"input": proj1_t.gelu},
        request_state_outputs={at_tensors.PROJ2_OUTPUT},
    )

    # Text (non-layered)
    embed_tokens_t = text_tensors.create_embed_tokens("spike.text.embed")
    audio_inject_t = text_tensors.create_audio_inject(
        "spike.text.audio_inject",
        bindings={
            "audio_features": proj2_t.linear,
            "index_copy": embed_tokens_t.embedding,
        },
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

    # Text layers (layered)
    text_layer_ts = []
    text_hidden = audio_inject_t.index_copy
    for layer_idx in range(tc.num_hidden_layers):
        bindings = {
            "hidden_states": text_hidden,
            "index_copy": key_caches[layer_idx],
            "index_copy_1": value_caches[layer_idx],
        }
        if layer_idx > 0:
            bindings["cache_position"] = text_layer_ts[0].cache_position
            bindings["position_embeddings_0"] = text_layer_ts[0].position_embeddings_0
            bindings["position_embeddings_1"] = text_layer_ts[0].position_embeddings_1
        layer_tensors = text_layer_tensors.create_text_layer(
            f"spike.text.layer.{layer_idx}",
            layer_idx=layer_idx,
            bindings=bindings,
        )
        text_layer_ts.append(layer_tensors)
        text_hidden = layer_tensors.add_7
    text_norm_t = text_tensors.create_text_norm(
        "spike.text.norm",
        bindings={"hidden_states": text_layer_ts[-1].add_7},
    )
    lm_head_t = text_tensors.create_lm_head(
        "spike.text.lm_head",
        bindings={"input": text_norm_t.mul_1},
        request_state_outputs={text_tensors.LM_HEAD_OUTPUT},
    )

    # Decode (non-layered, reuse for all decode steps)
    decode_embed_t = decode_tensors.create_decode_embed("spike.decode.embed")

    # Decode layers (layered)
    decode_layer_ts = []
    decode_hidden = decode_embed_t.embedding
    for layer_idx in range(tc.num_hidden_layers):
        bindings = {
            "hidden_states": decode_hidden,
            "index_copy": key_caches[layer_idx],
            "index_copy_1": value_caches[layer_idx],
        }
        if layer_idx > 0:
            bindings["cache_position"] = decode_layer_ts[0].cache_position
            bindings["attention_mask"] = decode_layer_ts[0].attention_mask
            bindings["position_embeddings_0"] = decode_layer_ts[0].position_embeddings_0
            bindings["position_embeddings_1"] = decode_layer_ts[0].position_embeddings_1
        layer_tensors = decode_layer_tensors.create_decode_layer(
            f"spike.decode.layer.{layer_idx}",
            layer_idx=layer_idx,
            bindings=bindings,
        )
        decode_layer_ts.append(layer_tensors)
        decode_hidden = layer_tensors.add_7
    decode_norm_t = decode_tensors.create_decode_norm(
        "spike.decode.norm",
        bindings={"hidden_states": decode_layer_ts[-1].add_7},
    )
    decode_lm_head_t = decode_tensors.create_decode_lm_head(
        "spike.decode.lm_head",
        bindings={"input": decode_norm_t.mul_1},
        request_state_outputs={decode_tensors.DECODE_LM_HEAD_OUTPUT},
    )

    # Materialize all weights to persistent GPU memory
    for t in [conv2d1_t, conv2d2_t, conv2d3_t, conv_out_t, ln_post_t, proj1_t, proj2_t]:
        _materialize_weights(rt, t, weights)
    for t in encoder_layer_ts:
        _materialize_weights(rt, t, weights)
    _materialize_weights(rt, embed_tokens_t, weights)
    _materialize_weights(rt, text_norm_t, weights)
    _materialize_weights(rt, lm_head_t, weights)
    for t in text_layer_ts:
        _materialize_weights(rt, t, weights)
    _materialize_weights(rt, decode_embed_t, weights)
    _materialize_weights(rt, decode_norm_t, weights)
    _materialize_weights(rt, decode_lm_head_t, weights)
    for t in decode_layer_ts:
        _materialize_weights(rt, t, weights)
    zero_cache = np.zeros(
        (1, tc.num_key_value_heads, max_sequence_length, tc.head_dim),
        dtype=np.float32,
    )
    rt.initialize_request_state(
        {cache: zero_cache for cache in key_caches + value_caches}
    )
    print("  Weights materialized.")

    # === Audio Tower ===
    print("\n=== Phase 1: Audio Tower ===")
    feat_len = prepared.audio_feature_length
    input_features = np.ascontiguousarray(prepared.input_features[0, :, :feat_len], dtype=np.float32)

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
    conv_out_shape = tuple(int(dim) for dim in conv_out_t.linear.spec.shape)
    _, t_dim, _ = conv_out_shape
    pos_emb = _compute_positional_embedding(t_dim, ac.d_model)
    position_embedding = np.ascontiguousarray(
        np.broadcast_to(pos_emb[None, :t_dim, :], conv_out_shape),
        dtype=np.float32,
    )

    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    valid_positions = []
    for i, fl in enumerate(feature_lens_after_cnn):
        for j in range(int(fl)):
            valid_positions.append(i * t_dim + j)
    compact_index = np.array(valid_positions, dtype=np.int32)
    print(f"  hidden_states after compact: {audio_position_compact_t.index_select.spec.shape}")

    # cu_seqlens
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

    # Build block-diagonal attention mask from cu_seqlens
    seq_len = compact_index.shape[0]
    attention_mask = np.full((1, 1, seq_len, seq_len), -np.finfo(np.float32).max, dtype=np.float32)
    for i in range(1, len(cu_seqlens)):
        s, e = int(cu_seqlens[i - 1]), int(cu_seqlens[i])
        attention_mask[0, 0, s:e, s:e] = 0.0

    # Conv layers + compact stay on GPU; only host-prepared indices/position are uploaded.
    print(f"  conv stack ({padded_feature.shape})...")
    rt._inputs.clear()
    rt.register_inputs(
        {
            conv2d1_t.x: padded_feature,
            audio_position_compact_t.position_embedding: position_embedding,
            audio_position_compact_t.compact_index: compact_index,
        }
    )
    with rt.frame("spike.audio.conv_stack"):
        dispatch.run_conv2d1(rt, conv2d1_t)
        dispatch.run_conv2d2(rt, conv2d2_t)
        dispatch.run_conv2d3(rt, conv2d3_t)
        dispatch.run_conv_out(rt, conv_out_t)
        dispatch.run_audio_position_compact(rt, audio_position_compact_t)
    _require_gpu_output(audio_position_compact_t.index_select)

    # Encoder layers + projection stay in one GPU frame.
    print("  encoder + projection...")
    rt._inputs.clear()
    rt.register_inputs(
        {
            encoder_layer_ts[0].attention_mask: attention_mask,
        }
    )
    with rt.frame("spike.audio.encoder_project"):
        for layer_idx, layer_tensors in enumerate(encoder_layer_ts):
            dispatch.run_encoder_layer(rt, layer_tensors)
            if layer_idx % 6 == 5:
                print(f"    layer {layer_idx} done")
        dispatch.run_ln_post(rt, ln_post_t)
        dispatch.run_proj1(rt, proj1_t)
        dispatch.run_proj2(rt, proj2_t)
    audio_hidden_t = proj2_t.linear
    _require_gpu_output(audio_hidden_t)
    print(f"  Audio tower output: {audio_hidden_t.spec.shape}")

    # === Text Prefill ===
    print("\n=== Phase 2: Text Prefill ===")
    prompt_length = prepared.prompt_length

    ids_flat = prepared.input_ids.flatten()
    audio_positions = np.where(ids_flat == 151676)[0].astype(np.int32)
    if len(audio_positions) > 0:
        audio_start = int(audio_positions[0])
        audio_end = audio_start + len(audio_positions)
        print(f"    Injecting audio [{audio_start}:{audio_end}]")

    rope_cos, rope_sin = _compute_rope(prompt_length, tc.head_dim)

    # Embedding, audio injection, and decoder layers stay on GPU.
    print(f"  embed + audio inject + decoder layers x {tc.num_hidden_layers}...")
    rt._inputs.clear()
    rt.register_inputs(
        {
            embed_tokens_t.input: prepared.input_ids.astype(np.int32),
            audio_inject_t.audio_positions: audio_positions,
            text_layer_ts[0].cache_position: np.arange(prompt_length, dtype=np.int32),
            text_layer_ts[0].position_embeddings_0: rope_cos,
            text_layer_ts[0].position_embeddings_1: rope_sin,
        }
    )
    with rt.frame("spike.text.prefill"):
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
    rt.register_inputs({eos_token_ids_t: eos_token_array})
    first_token = _run_token_select(
        rt,
        logits=logits_t,
        eos_token_ids=eos_token_ids_t,
        next_token=next_token_t,
        done=done_t,
        frame_name="spike.text.token_select",
    )
    print(f"  First token: {first_token}")

    # === Decode Loop ===
    print("\n=== Phase 3: Decode Loop ===")
    eos_token_set = set(eos_token_ids)
    generated_tokens = [first_token]

    for step in range(max_new_tokens - 1):
        if generated_tokens[-1] in eos_token_set:
            print(f"  EOS at step {step}")
            break

        cache_pos = prompt_length + step
        token_input = np.array([[generated_tokens[-1]]], dtype=np.int32)
        decode_attention_mask = np.full(
            (1, 1, 1, max_sequence_length),
            -np.finfo(np.float32).max,
            dtype=np.float32,
        )
        decode_attention_mask[0, 0, 0, : cache_pos + 1] = 0.0

        rope_cos, rope_sin = _compute_rope(1, tc.head_dim, start_pos=cache_pos)

        rt._inputs.clear()
        rt.register_inputs(
            {
                decode_embed_t.input: token_input,
                decode_layer_ts[0].cache_position: np.array([cache_pos], dtype=np.int32),
                decode_layer_ts[0].attention_mask: decode_attention_mask,
                decode_layer_ts[0].position_embeddings_0: rope_cos,
                decode_layer_ts[0].position_embeddings_1: rope_sin,
            }
        )
        with rt.frame(f"spike.decode.{step:04d}"):
            dispatch.run_decode_embed(rt, decode_embed_t)
            for layer_tensors in decode_layer_ts:
                dispatch.run_decode_layer(rt, layer_tensors)
            dispatch.run_decode_norm(rt, decode_norm_t)
            dispatch.run_decode_lm_head(rt, decode_lm_head_t)

        decode_logits_t = decode_lm_head_t.linear
        _require_gpu_output(decode_logits_t)
        rt.register_inputs({eos_token_ids_t: eos_token_array})
        next_token = _run_token_select(
            rt,
            logits=decode_logits_t,
            eos_token_ids=eos_token_ids_t,
            next_token=next_token_t,
            done=done_t,
            frame_name=f"spike.decode.token_select.{step:04d}",
        )
        generated_tokens.append(next_token)

        if step < 5 or step % 20 == 0:
            print(f"  Step {step}: token={next_token}")

    # Decode text
    print("\n=== Result ===")
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
