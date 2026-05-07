"""Full ASR pipeline using generated shaders and dispatch functions.

Run from project root:
    .venv/bin/python -m models.spike_qwen3_asr.run
"""

from __future__ import annotations

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
from torch2vk.runtime.session import RuntimeSession


def _run(rt, dispatch_fn, tensors_obj, weight_map, output_field, weights, layer_idx, user_inputs, frame_name,
         *, pytorch_model_submodule=None, pytorch_args=(), pytorch_kwargs=None):
    rt._inputs.clear()
    for field, key_tmpl in weight_map.items():
        key = key_tmpl.format(i=layer_idx)
        rt.register_inputs({getattr(tensors_obj, field): weights.get_tensor(key).float().numpy()})
    for field, data in user_inputs.items():
        rt.register_inputs({getattr(tensors_obj, field): data})

    frame_kw = {}
    if pytorch_model_submodule is not None:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
        frame_kw["pytorch_model_class"] = Qwen3ASRForConditionalGeneration
        frame_kw["pytorch_model_submodule"] = pytorch_model_submodule.format(i=layer_idx)
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


def _gelu_numpy(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))).astype(np.float32)


def _compute_rope(seq_len: int, head_dim: int, rope_theta: float = 5_000_000.0, start_pos: int = 0) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(start_pos, start_pos + seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32).reshape(1, seq_len, head_dim)
    sin = np.sin(emb).astype(np.float32).reshape(1, seq_len, head_dim)
    return cos, sin


# ==============================================================
# Main pipeline
# ==============================================================

def main() -> int:
    wav_path = Path("tests/fixtures/qwen3_asr_asknot.wav")
    if not wav_path.exists():
        print(f"ERROR: Test wav not found at {wav_path}")
        return 1

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
    t = at_tensors.create_conv2d1("spike.audio.conv2d1")
    conv1_out = _gelu_numpy(_run(rt, dispatch.run_conv2d1, t, at_tensors.CONV2D1_WEIGHT_MAP, at_tensors.CONV2D1_OUTPUT, weights, 0, {"input": padded_feature}, "spike.audio.conv2d1"))

    print(f"  conv2d2 ({conv1_out.shape})...")
    t = at_tensors.create_conv2d2("spike.audio.conv2d2")
    conv2_out = _gelu_numpy(_run(rt, dispatch.run_conv2d2, t, at_tensors.CONV2D2_WEIGHT_MAP, at_tensors.CONV2D2_OUTPUT, weights, 0, {"input": conv1_out}, "spike.audio.conv2d2"))

    print(f"  conv2d3 ({conv2_out.shape})...")
    t = at_tensors.create_conv2d3("spike.audio.conv2d3")
    conv3_out = _gelu_numpy(_run(rt, dispatch.run_conv2d3, t, at_tensors.CONV2D3_WEIGHT_MAP, at_tensors.CONV2D3_OUTPUT, weights, 0, {"input": conv2_out}, "spike.audio.conv2d3"))

    # Reshape + conv_out
    b, c, f, t_dim = conv3_out.shape
    reshaped = conv3_out.transpose(0, 3, 1, 2).reshape(b, t_dim, c * f)
    conv_out_input = reshaped.reshape(b * t_dim, c * f)
    print(f"  conv_out ({conv_out_input.shape})...")
    t = at_tensors.create_conv_out("spike.audio.conv_out")
    conv_out_result = _run(rt, dispatch.run_conv_out, t, at_tensors.CONV_OUT_WEIGHT_MAP, at_tensors.CONV_OUT_OUTPUT, weights, 0, {"input": conv_out_input}, "spike.audio.conv_out")
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
        t = enc_tensors.create_encoder_layer(f"spike.audio.enc.{layer_idx}")
        hidden_states = _run(rt, dispatch.run_encoder_layer, t, enc_tensors.ENCODER_LAYER_WEIGHT_MAP, enc_tensors.ENCODER_LAYER_OUTPUT, weights, layer_idx, {"hidden_states": hidden_states, "cu_seqlens": cu_seqlens, "attention_mask": attention_mask}, f"spike.audio.enc.{layer_idx}")
        if layer_idx == 0:
            print(f"    layer 0: max={hidden_states.max():.4f}")
        if layer_idx % 6 == 5:
            print(f"    layer {layer_idx} done")

    # ln_post, proj1, proj2
    print("  ln_post...")
    t = at_tensors.create_ln_post("spike.audio.ln_post")
    hidden_states = _run(rt, dispatch.run_ln_post, t, at_tensors.LN_POST_WEIGHT_MAP, at_tensors.LN_POST_OUTPUT, weights, 0, {"input": hidden_states}, "spike.audio.ln_post")

    print("  proj1 + gelu...")
    t = at_tensors.create_proj1("spike.audio.proj1")
    hidden_states = _gelu_numpy(_run(rt, dispatch.run_proj1, t, at_tensors.PROJ1_WEIGHT_MAP, at_tensors.PROJ1_OUTPUT, weights, 0, {"input": hidden_states}, "spike.audio.proj1"))

    print("  proj2...")
    t = at_tensors.create_proj2("spike.audio.proj2")
    audio_hidden = _run(rt, dispatch.run_proj2, t, at_tensors.PROJ2_WEIGHT_MAP, at_tensors.PROJ2_OUTPUT, weights, 0, {"input": hidden_states}, "spike.audio.proj2")
    print(f"  Audio tower output: {audio_hidden.shape}")

    # === Text Prefill ===
    print("\n=== Phase 2: Text Prefill ===")
    prompt_length = prepared.prompt_length

    print("  embed_tokens...")
    t = text_tensors.create_embed_tokens("spike.text.embed")
    embedded = _run(rt, dispatch.run_embed_tokens, t, text_tensors.EMBED_TOKENS_WEIGHT_MAP, text_tensors.EMBED_TOKENS_OUTPUT, weights, 0, {"input": prepared.input_ids.astype(np.int32)}, "spike.text.embed")

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
    print(f"  Decoder layers × {tc.num_hidden_layers}...")
    for layer_idx in range(tc.num_hidden_layers):
        t = text_layer_tensors.create_text_layer(f"spike.text.layer.{layer_idx}")
        hidden_states = _run(
            rt, dispatch.run_text_layer, t,
            text_layer_tensors.TEXT_LAYER_WEIGHT_MAP, text_layer_tensors.TEXT_LAYER_OUTPUT,
            weights, layer_idx,
            {"hidden_states": hidden_states, "position_embeddings_0": rope_cos, "position_embeddings_1": rope_sin},
            f"spike.text.layer.{layer_idx}",
        )
        if layer_idx == 0:
            print(f"    layer 0: max={hidden_states.max():.4f}")
        if layer_idx % 7 == 6:
            print(f"    layer {layer_idx} done")

    print("  final_norm...")
    t = text_tensors.create_text_norm("spike.text.norm")
    normed = _run(rt, dispatch.run_text_norm, t, text_tensors.TEXT_NORM_WEIGHT_MAP, text_tensors.TEXT_NORM_OUTPUT, weights, 0, {"hidden_states": hidden_states}, "spike.text.norm")

    print("  lm_head...")
    last_hidden = normed[:, -1:, :]
    t = text_tensors.create_lm_head("spike.text.lm_head")
    logits = _run(rt, dispatch.run_lm_head, t, text_tensors.LM_HEAD_WEIGHT_MAP, text_tensors.LM_HEAD_OUTPUT, weights, 0, {"input": last_hidden}, "spike.text.lm_head")
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

        t = decode_tensors.create_decode_embed(f"spike.decode.embed.{step}")
        hidden = _run(rt, dispatch.run_decode_embed, t, decode_tensors.DECODE_EMBED_WEIGHT_MAP, decode_tensors.DECODE_EMBED_OUTPUT, weights, 0, {"input": token_input}, f"spike.decode.embed.{step}")

        rope_cos, rope_sin = _compute_rope(1, tc.head_dim, start_pos=cache_pos)

        for layer_idx in range(tc.num_hidden_layers):
            t = decode_layer_tensors.create_decode_layer(f"spike.decode.layer.{step}.{layer_idx}")
            hidden = _run(rt, dispatch.run_decode_layer, t, decode_layer_tensors.DECODE_LAYER_WEIGHT_MAP, decode_layer_tensors.DECODE_LAYER_OUTPUT, weights, layer_idx, {"hidden_states": hidden, "position_embeddings_0": rope_cos, "position_embeddings_1": rope_sin}, f"spike.decode.layer.{step}.{layer_idx}")

        t = decode_tensors.create_decode_norm(f"spike.decode.norm.{step}")
        normed = _run(rt, dispatch.run_decode_norm, t, decode_tensors.DECODE_NORM_WEIGHT_MAP, decode_tensors.DECODE_NORM_OUTPUT, weights, 0, {"hidden_states": hidden}, f"spike.decode.norm.{step}")

        t = decode_tensors.create_decode_lm_head(f"spike.decode.lm_head.{step}")
        logits = _run(rt, dispatch.run_decode_lm_head, t, decode_tensors.DECODE_LM_HEAD_WEIGHT_MAP, decode_tensors.DECODE_LM_HEAD_OUTPUT, weights, 0, {"input": normed}, f"spike.decode.lm_head.{step}")
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

    # Print comparison results
    if rt.compare_results:
        print(f"\n=== PyTorch Comparison ===")
        for r in rt.compare_results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.frame}: max_abs={r.max_abs:.6f}")

    rt.close()
    return text


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result else 1)
