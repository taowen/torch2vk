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
from safetensors import safe_open

from models.hf_cache import resolve_cached_model
from models.qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.qwen3_asr.pytorch.example import REPO_ID
from models.spike_qwen3_asr import dispatch
from models.spike_qwen3_asr.tensors import audio_tower as at_tensors
from models.spike_qwen3_asr.tensors import decode as decode_tensors
from models.spike_qwen3_asr.tensors import decode_layer as decode_layer_tensors
from models.spike_qwen3_asr.tensors import encoder_layer as enc_tensors
from models.spike_qwen3_asr.tensors import text as text_tensors
from models.spike_qwen3_asr.tensors import text_layer as text_layer_tensors
from torch2vk.runtime.logical import LogicalTensor, TensorRole
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


def _run(rt, dispatch_fn, tensors_obj, output_field, user_inputs, frame_name,
         *, pytorch_model_submodule=None, pytorch_args=(), pytorch_kwargs=None):
    rt._inputs.clear()
    for field, data in user_inputs.items():
        rt.register_inputs({getattr(tensors_obj, field): data})

    frame_kw = {}
    if pytorch_model_submodule is not None:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
        frame_kw["pytorch_model_class"] = Qwen3ASRForConditionalGeneration
        frame_kw["pytorch_model_submodule"] = pytorch_model_submodule
        if pytorch_args:
            frame_kw["pytorch_args"] = tuple(getattr(tensors_obj, f) for f in pytorch_args)
        if pytorch_kwargs:
            resolved_kwargs = {}
            for k, v in pytorch_kwargs.items():
                if isinstance(v, tuple):
                    resolved_kwargs[k] = tuple(getattr(tensors_obj, f) for f in v)
                else:
                    resolved_kwargs[k] = getattr(tensors_obj, v)
            frame_kw["pytorch_kwargs"] = resolved_kwargs

    with rt.frame(frame_name, **frame_kw):
        dispatch_fn(rt, tensors_obj)
        output_t = getattr(tensors_obj, output_field)
        return rt.device.readback_tensor(spec=output_t.spec, slice=output_t.buffer)


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
    os.dup2(devnull.fileno(), 1); os.dup2(devnull.fileno(), 2)
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
        config = Qwen3ASRConfig(**json.loads(config_payload))
    finally:
        os.dup2(stdout_fd, 1); os.dup2(stderr_fd, 2)
        os.close(stdout_fd); os.close(stderr_fd); devnull.close()

    ac = config.thinker_config.audio_config
    tc = config.thinker_config.text_config

    processor, prepared = prepare_qwen3_asr_inputs(model_dir=model_dir, wav=str(wav_path))
    weights = safe_open(str(Path(model_dir) / "model.safetensors"), framework="pt", device="cpu")
    rt = RuntimeSession.open(device_index=0, model_dir=model_dir)

    # === Create all tensor objects upfront and materialize weights ===
    print("Materializing weights...")

    # Audio tower (non-layered)
    conv2d1_t = at_tensors.create_conv2d1("spike.audio.conv2d1")
    conv2d2_t = at_tensors.create_conv2d2("spike.audio.conv2d2")
    conv2d3_t = at_tensors.create_conv2d3("spike.audio.conv2d3")
    conv_out_t = at_tensors.create_conv_out("spike.audio.conv_out")
    ln_post_t = at_tensors.create_ln_post("spike.audio.ln_post")
    proj1_t = at_tensors.create_proj1("spike.audio.proj1")
    proj2_t = at_tensors.create_proj2("spike.audio.proj2")

    # Encoder layers (layered)
    encoder_layer_ts = [enc_tensors.create_encoder_layer(f"spike.audio.enc.{i}", layer_idx=i) for i in range(ac.encoder_layers)]

    # Text (non-layered)
    embed_tokens_t = text_tensors.create_embed_tokens("spike.text.embed")
    text_norm_t = text_tensors.create_text_norm("spike.text.norm")
    lm_head_t = text_tensors.create_lm_head("spike.text.lm_head")

    # Text layers (layered)
    text_layer_ts = [text_layer_tensors.create_text_layer(f"spike.text.layer.{i}", layer_idx=i) for i in range(tc.num_hidden_layers)]

    # Decode (non-layered, reuse for all decode steps)
    decode_embed_t = decode_tensors.create_decode_embed("spike.decode.embed")
    decode_norm_t = decode_tensors.create_decode_norm("spike.decode.norm")
    decode_lm_head_t = decode_tensors.create_decode_lm_head("spike.decode.lm_head")

    # Decode layers (layered)
    decode_layer_ts = [decode_layer_tensors.create_decode_layer(f"spike.decode.layer.{i}", layer_idx=i) for i in range(tc.num_hidden_layers)]

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

    # Conv layers
    print(f"  conv2d1 ({padded_feature.shape})...")
    conv1_out = _run(rt, dispatch.run_conv2d1, conv2d1_t, at_tensors.CONV2D1_OUTPUT, {"x": padded_feature}, "spike.audio.conv2d1")

    print(f"  conv2d2 ({conv1_out.shape})...")
    conv2_out = _run(rt, dispatch.run_conv2d2, conv2d2_t, at_tensors.CONV2D2_OUTPUT, {"x": conv1_out}, "spike.audio.conv2d2")

    print(f"  conv2d3 ({conv2_out.shape})...")
    conv3_out = _run(rt, dispatch.run_conv2d3, conv2d3_t, at_tensors.CONV2D3_OUTPUT, {"x": conv2_out}, "spike.audio.conv2d3")

    # Reshape + conv_out
    b, c, f, t_dim = conv3_out.shape
    reshaped = conv3_out.transpose(0, 3, 1, 2).reshape(b, t_dim, c * f)
    conv_out_input = reshaped.reshape(b * t_dim, c * f)
    print(f"  conv_out ({conv_out_input.shape})...")
    conv_out_result = _run(rt, dispatch.run_conv_out, conv_out_t, at_tensors.CONV_OUT_OUTPUT, {"input": conv_out_input}, "spike.audio.conv_out")
    conv_out_result = conv_out_result.reshape(b, t_dim, ac.d_model)

    # Positional embedding + compact
    pos_emb = _compute_positional_embedding(t_dim, ac.d_model)
    padded_embed = conv_out_result + pos_emb[None, :t_dim, :]

    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    valid_positions = []
    for i, fl in enumerate(feature_lens_after_cnn):
        for j in range(int(fl)):
            valid_positions.append((i, j))
    hidden_states = np.array([padded_embed[i, j] for i, j in valid_positions], dtype=np.float32)
    print(f"  hidden_states after compact: {hidden_states.shape}")

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
    seq_len = hidden_states.shape[0]
    attention_mask = np.full((1, 1, seq_len, seq_len), -np.finfo(np.float32).max, dtype=np.float32)
    for i in range(1, len(cu_seqlens)):
        s, e = int(cu_seqlens[i - 1]), int(cu_seqlens[i])
        attention_mask[0, 0, s:e, s:e] = 0.0

    # Encoder layers
    for layer_idx in range(ac.encoder_layers):
        hidden_states = _run(rt, dispatch.run_encoder_layer, encoder_layer_ts[layer_idx], enc_tensors.ENCODER_LAYER_OUTPUT, {"hidden_states": hidden_states, "cu_seqlens": cu_seqlens, "attention_mask": attention_mask}, f"spike.audio.enc.{layer_idx}")
        if layer_idx == 0:
            print(f"    layer 0: max={hidden_states.max():.4f}")
        if layer_idx % 6 == 5:
            print(f"    layer {layer_idx} done")

    # ln_post, proj1, proj2
    print("  ln_post...")
    hidden_states = _run(rt, dispatch.run_ln_post, ln_post_t, at_tensors.LN_POST_OUTPUT, {"input": hidden_states}, "spike.audio.ln_post")

    print("  proj1 + gelu...")
    hidden_states = _run(rt, dispatch.run_proj1, proj1_t, at_tensors.PROJ1_OUTPUT, {"x": hidden_states}, "spike.audio.proj1")

    print("  proj2...")
    audio_hidden = _run(rt, dispatch.run_proj2, proj2_t, at_tensors.PROJ2_OUTPUT, {"input": hidden_states}, "spike.audio.proj2")
    print(f"  Audio tower output: {audio_hidden.shape}")

    # === Text Prefill ===
    print("\n=== Phase 2: Text Prefill ===")
    prompt_length = prepared.prompt_length

    print("  embed_tokens...")
    embedded = _run(rt, dispatch.run_embed_tokens, embed_tokens_t, text_tensors.EMBED_TOKENS_OUTPUT, {"input": prepared.input_ids.astype(np.int32)}, "spike.text.embed")

    # Inject audio
    ids_flat = prepared.input_ids.flatten()
    audio_positions = np.where(ids_flat == 151646)[0]
    if len(audio_positions) > 0:
        audio_start = int(audio_positions[0])
        audio_end = audio_start + audio_hidden.shape[0]
        print(f"    Injecting audio [{audio_start}:{audio_end}]")
        embedded[0, audio_start:audio_end, :] = audio_hidden

    rope_cos, rope_sin = _compute_rope(prompt_length, tc.head_dim)

    # Decoder layers
    hidden_states = embedded
    print(f"  Decoder layers x {tc.num_hidden_layers}...")
    for layer_idx in range(tc.num_hidden_layers):
        hidden_states = _run(
            rt, dispatch.run_text_layer, text_layer_ts[layer_idx],
            text_layer_tensors.TEXT_LAYER_OUTPUT,
            {"hidden_states": hidden_states, "position_embeddings_0": rope_cos, "position_embeddings_1": rope_sin},
            f"spike.text.layer.{layer_idx}",
        )
        if layer_idx == 0:
            print(f"    layer 0: max={hidden_states.max():.4f}")
        if layer_idx % 7 == 6:
            print(f"    layer {layer_idx} done")

    print("  final_norm...")
    normed = _run(rt, dispatch.run_text_norm, text_norm_t, text_tensors.TEXT_NORM_OUTPUT, {"hidden_states": hidden_states}, "spike.text.norm")

    print("  lm_head...")
    last_hidden = normed[:, -1:, :]
    logits = _run(rt, dispatch.run_lm_head, lm_head_t, text_tensors.LM_HEAD_OUTPUT, {"input": last_hidden}, "spike.text.lm_head")
    first_token = int(np.argmax(logits[0, -1, :]))
    print(f"  First token: {first_token}")

    # === Decode Loop ===
    print("\n=== Phase 3: Decode Loop ===")
    max_new_tokens = 20
    eos_token_ids = {151645, 151643}
    generated_tokens = [first_token]

    for step in range(max_new_tokens - 1):
        if generated_tokens[-1] in eos_token_ids:
            print(f"  EOS at step {step}")
            break

        cache_pos = prompt_length + step
        token_input = np.array([[generated_tokens[-1]]], dtype=np.int32)

        hidden = _run(rt, dispatch.run_decode_embed, decode_embed_t, decode_tensors.DECODE_EMBED_OUTPUT, {"input": token_input}, "spike.decode.embed")

        rope_cos, rope_sin = _compute_rope(1, tc.head_dim, start_pos=cache_pos)

        for layer_idx in range(tc.num_hidden_layers):
            hidden = _run(rt, dispatch.run_decode_layer, decode_layer_ts[layer_idx], decode_layer_tensors.DECODE_LAYER_OUTPUT, {"hidden_states": hidden, "position_embeddings_0": rope_cos, "position_embeddings_1": rope_sin}, f"spike.decode.layer.{layer_idx}")

        normed = _run(rt, dispatch.run_decode_norm, decode_norm_t, decode_tensors.DECODE_NORM_OUTPUT, {"hidden_states": hidden}, "spike.decode.norm")

        logits = _run(rt, dispatch.run_decode_lm_head, decode_lm_head_t, decode_tensors.DECODE_LM_HEAD_OUTPUT, {"input": normed}, "spike.decode.lm_head")
        next_token = int(np.argmax(logits[0, -1, :]))
        generated_tokens.append(next_token)

        if step < 5 or step % 20 == 0:
            print(f"  Step {step}: token={next_token}")

    # Decode text
    print(f"\n=== Result ===")
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
