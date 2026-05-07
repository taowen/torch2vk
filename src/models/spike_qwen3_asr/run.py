"""Full ASR pipeline using generated shaders and dispatch functions.

Run from project root:
    .venv/bin/python -m models.spike_qwen3_asr.run
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

from models.hf_cache import resolve_cached_model
from models.qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.qwen3_asr.pytorch.example import REPO_ID
from models.spike_qwen3_asr import dispatch
from torch2vk.runtime.logical import LogicalTensor, MemoryClass, TensorLifetime, TensorRole
from torch2vk.runtime.session import RuntimeSession
from torch2vk.vulkan.types import TensorSpec


def _build_tensors(plan_meta: dict) -> dict[str, LogicalTensor]:
    tensors = {}
    for name, meta in plan_meta["tensors"].items():
        shape = tuple(meta["shape"])
        dtype = "int32" if meta["dtype"] in ("int64", "int32") else "float32"
        if meta["kind"] == "parameter":
            role, memory, lifetime = TensorRole.INPUT, MemoryClass.HOST_INPUT, TensorLifetime.FRAME
        elif meta["kind"] == "user_input":
            role, memory, lifetime = TensorRole.INPUT, MemoryClass.HOST_INPUT, TensorLifetime.FRAME
        else:
            role, memory, lifetime = TensorRole.ACTIVATION, MemoryClass.FRAME_WORKSPACE, TensorLifetime.FRAME
        tensors[name] = LogicalTensor(name=name, spec=TensorSpec(dtype=dtype, shape=shape), role=role, memory=memory, lifetime=lifetime)
    return tensors


def _run_submodule(
    rt: RuntimeSession,
    dispatch_fn,
    plan_meta: dict,
    weights_handle,
    weight_prefix: str,
    user_inputs: dict[str, np.ndarray],
    frame_name: str,
) -> np.ndarray:
    rt._inputs.clear()
    tensors = _build_tensors(plan_meta)

    weight_map = {}
    for tensor_name, safetensors_key in plan_meta["param_map"].items():
        weight_map[tensors[tensor_name]] = weights_handle.get_tensor(safetensors_key).float().numpy()
    rt.register_inputs(weight_map)

    input_map = {}
    for name, data in user_inputs.items():
        if name in tensors:
            input_map[tensors[name]] = data
    rt.register_inputs(input_map)

    with rt.frame(frame_name):
        dispatch_fn(rt, tensors)
        output_t = tensors[plan_meta["output"]]
        return rt.device.readback_tensor(spec=output_t.spec, slice=output_t.buffer)


def _run_submodule_reprefix(
    rt: RuntimeSession,
    dispatch_fn,
    plan_meta: dict,
    weights_handle,
    actual_weight_prefix: str,
    user_inputs: dict[str, np.ndarray],
    frame_name: str,
) -> np.ndarray:
    """Like _run_submodule but remaps weight keys to a different layer prefix."""
    rt._inputs.clear()
    tensors = _build_tensors(plan_meta)

    # Find the common prefix in the stored param_map (e.g. "thinker.audio_tower.layers.0.")
    first_key = next(iter(plan_meta["param_map"].values()))
    # The stored prefix is everything up to and including the layer index dot
    # e.g. "thinker.model.layers.0." or "thinker.audio_tower.layers.0."
    stored_prefix = first_key[:len(actual_weight_prefix)]
    # Find the actual stored prefix by finding common prefix of all keys
    for key in plan_meta["param_map"].values():
        while not key.startswith(stored_prefix):
            stored_prefix = stored_prefix[:-1]

    weight_map = {}
    for tensor_name, original_key in plan_meta["param_map"].items():
        actual_key = actual_weight_prefix + original_key[len(stored_prefix):]
        weight_map[tensors[tensor_name]] = weights_handle.get_tensor(actual_key).float().numpy()
    rt.register_inputs(weight_map)

    input_map = {}
    for name, data in user_inputs.items():
        if name in tensors:
            input_map[tensors[name]] = data
    rt.register_inputs(input_map)

    with rt.frame(frame_name):
        dispatch_fn(rt, tensors)
        output_t = tensors[plan_meta["output"]]
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

    # Load plans metadata
    spike_dir = Path(__file__).parent
    plans = json.loads((spike_dir / "plans.json").read_text())

    print("Preparing inputs...")
    model_dir = resolve_cached_model(REPO_ID)
    config_payload = json.loads((Path(model_dir) / "config.json").read_text())

    devnull = open(os.devnull, "w")
    stdout_fd, stderr_fd = os.dup(1), os.dup(2)
    os.dup2(devnull.fileno(), 1); os.dup2(devnull.fileno(), 2)
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
        config = Qwen3ASRConfig(**config_payload)
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
    conv1_out = _gelu_numpy(_run_submodule(
        rt, dispatch.run_conv2d1, plans["run_conv2d1"], weights,
        "thinker.audio_tower.conv2d1.", {plans["run_conv2d1"]["user_inputs"][0]: padded_feature}, "spike.audio.conv2d1"))

    print(f"  conv2d2 ({conv1_out.shape})...")
    conv2_out = _gelu_numpy(_run_submodule(
        rt, dispatch.run_conv2d2, plans["run_conv2d2"], weights,
        "thinker.audio_tower.conv2d2.", {plans["run_conv2d2"]["user_inputs"][0]: conv1_out}, "spike.audio.conv2d2"))

    print(f"  conv2d3 ({conv2_out.shape})...")
    conv3_out = _gelu_numpy(_run_submodule(
        rt, dispatch.run_conv2d3, plans["run_conv2d3"], weights,
        "thinker.audio_tower.conv2d3.", {plans["run_conv2d3"]["user_inputs"][0]: conv2_out}, "spike.audio.conv2d3"))

    # Reshape + conv_out
    b, c, f, t = conv3_out.shape
    reshaped = conv3_out.transpose(0, 3, 1, 2).reshape(b, t, c * f)
    conv_out_input = reshaped.reshape(b * t, c * f)
    print(f"  conv_out ({conv_out_input.shape})...")
    conv_out_result = _run_submodule(
        rt, dispatch.run_conv_out, plans["run_conv_out"], weights,
        "thinker.audio_tower.conv_out.", {plans["run_conv_out"]["user_inputs"][0]: conv_out_input}, "spike.audio.conv_out")
    conv_out_result = conv_out_result.reshape(b, t, ac.d_model)

    # Positional embedding + compact
    pos_emb = _compute_positional_embedding(t, ac.d_model)
    padded_embed = conv_out_result + pos_emb[None, :t, :]

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

    # Encoder layers
    enc_plan = plans["run_encoder_layer"]
    enc_user_inputs = enc_plan["user_inputs"]
    for layer_idx in range(ac.encoder_layers):
        enc_inputs = {}
        for name in enc_user_inputs:
            t_meta = enc_plan["tensors"][name]
            if t_meta["dtype"] in ("int64", "int32"):
                enc_inputs[name] = cu_seqlens
            else:
                enc_inputs[name] = hidden_states
        hidden_states = _run_submodule_reprefix(
            rt, dispatch.run_encoder_layer, enc_plan, weights,
            f"thinker.audio_tower.layers.{layer_idx}.", enc_inputs, f"spike.audio.enc.{layer_idx}")
        if layer_idx == 0:
            print(f"    layer 0: max={hidden_states.max():.4f}")
        if layer_idx % 6 == 5:
            print(f"    layer {layer_idx} done")

    # ln_post, proj1, proj2
    print("  ln_post...")
    hidden_states = _run_submodule(
        rt, dispatch.run_ln_post, plans["run_ln_post"], weights,
        "thinker.audio_tower.ln_post.", {plans["run_ln_post"]["user_inputs"][0]: hidden_states}, "spike.audio.ln_post")

    print("  proj1 + gelu...")
    hidden_states = _gelu_numpy(_run_submodule(
        rt, dispatch.run_proj1, plans["run_proj1"], weights,
        "thinker.audio_tower.proj1.", {plans["run_proj1"]["user_inputs"][0]: hidden_states}, "spike.audio.proj1"))

    print("  proj2...")
    audio_hidden = _run_submodule(
        rt, dispatch.run_proj2, plans["run_proj2"], weights,
        "thinker.audio_tower.proj2.", {plans["run_proj2"]["user_inputs"][0]: hidden_states}, "spike.audio.proj2")
    print(f"  Audio tower output: {audio_hidden.shape}")

    # === Text Prefill ===
    print("\n=== Phase 2: Text Prefill ===")
    prompt_length = prepared.prompt_length

    print("  embed_tokens...")
    embedded = _run_submodule(
        rt, dispatch.run_embed_tokens, plans["run_embed_tokens"], weights,
        "thinker.model.embed_tokens.", {plans["run_embed_tokens"]["user_inputs"][0]: prepared.input_ids.astype(np.int32)},
        "spike.text.embed")

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
    text_plan = plans["run_text_layer"]
    text_user_inputs = text_plan["user_inputs"]
    hidden_states = embedded
    print(f"  Decoder layers × {tc.num_hidden_layers}...")
    for layer_idx in range(tc.num_hidden_layers):
        user_inputs = {}
        for name in text_user_inputs:
            if "hidden" in name:
                user_inputs[name] = hidden_states
            elif "position_embeddings_0" in name:
                user_inputs[name] = rope_cos
            elif "position_embeddings_1" in name:
                user_inputs[name] = rope_sin
        hidden_states = _run_submodule_reprefix(
            rt, dispatch.run_text_layer, text_plan, weights,
            f"thinker.model.layers.{layer_idx}.", user_inputs, f"spike.text.layer.{layer_idx}")
        if layer_idx == 0:
            print(f"    layer 0: max={hidden_states.max():.4f}")
        if layer_idx % 7 == 6:
            print(f"    layer {layer_idx} done")

    print("  final_norm...")
    normed = _run_submodule(
        rt, dispatch.run_text_norm, plans["run_text_norm"], weights,
        "thinker.model.norm.", {plans["run_text_norm"]["user_inputs"][0]: hidden_states}, "spike.text.norm")

    print("  lm_head...")
    last_hidden = normed[:, -1:, :]
    logits = _run_submodule(
        rt, dispatch.run_lm_head, plans["run_lm_head"], weights,
        "thinker.lm_head.", {plans["run_lm_head"]["user_inputs"][0]: last_hidden}, "spike.text.lm_head")
    first_token = int(np.argmax(logits[0, -1, :]))
    print(f"  First token: {first_token}")

    # === Decode Loop ===
    print("\n=== Phase 3: Decode Loop ===")
    max_new_tokens = 20
    eos_token_ids = {151645, 151643}
    generated_tokens = [first_token]

    decode_plan = plans["run_decode_layer"]
    decode_user_inputs = decode_plan["user_inputs"]

    for step in range(max_new_tokens - 1):
        if generated_tokens[-1] in eos_token_ids:
            print(f"  EOS at step {step}")
            break

        cache_pos = prompt_length + step
        token_input = np.array([[generated_tokens[-1]]], dtype=np.int32)

        hidden = _run_submodule(
            rt, dispatch.run_decode_embed, plans["run_decode_embed"], weights,
            "thinker.model.embed_tokens.", {plans["run_decode_embed"]["user_inputs"][0]: token_input},
            "spike.decode.embed")

        rope_cos, rope_sin = _compute_rope(1, tc.head_dim, start_pos=cache_pos)

        for layer_idx in range(tc.num_hidden_layers):
            user_inputs = {}
            for name in decode_user_inputs:
                if "hidden" in name:
                    user_inputs[name] = hidden
                elif "position_embeddings_0" in name:
                    user_inputs[name] = rope_cos
                elif "position_embeddings_1" in name:
                    user_inputs[name] = rope_sin
            hidden = _run_submodule_reprefix(
                rt, dispatch.run_decode_layer, decode_plan, weights,
                f"thinker.model.layers.{layer_idx}.", user_inputs, f"spike.decode.layer.{layer_idx}")

        normed = _run_submodule(
            rt, dispatch.run_decode_norm, plans["run_decode_norm"], weights,
            "thinker.model.norm.", {plans["run_decode_norm"]["user_inputs"][0]: hidden}, "spike.decode.norm")
        logits = _run_submodule(
            rt, dispatch.run_decode_lm_head, plans["run_decode_lm_head"], weights,
            "thinker.lm_head.", {plans["run_decode_lm_head"]["user_inputs"][0]: normed}, "spike.decode.lm_head")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
