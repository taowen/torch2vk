"""Full ASR pipeline: WAV → text, entirely on Vulkan compute shaders via torch2vk.export.

Every compute-intensive submodule is exported via torch2vk.export and dispatched on Vulkan.
Only data layout transformations (chunk/pad/reshape/compact) run on CPU/NumPy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gc

import numpy as np
import torch
from safetensors import safe_open
from torch.export.graph_signature import InputKind

from models.hf_cache import resolve_cached_model
from models.qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.qwen3_asr.pytorch.example import REPO_ID
from torch2vk.export import export_submodule
from torch2vk.export.graph import SKIP_OPS, is_alias_op, node_input_names
from torch2vk.export.registry import DEFAULT_REGISTRY
from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
)
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader import IOKind
from torch2vk.vulkan.types import TensorSpec


# ============================================================
# Model loading
# ============================================================

def _load_model_and_config():
    model_dir = resolve_cached_model(REPO_ID)
    payload = json.loads((Path(model_dir) / "config.json").read_text())
    devnull = open(os.devnull, "w")
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        config = Qwen3ASRConfig(**payload)
        with torch.device("meta"):
            model = Qwen3ASRForConditionalGeneration(config)
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        devnull.close()
    return model, config, Path(model_dir)


# ============================================================
# Export-based dispatch engine
# ============================================================

class ExportedSubmodule:
    """Export a submodule once, dispatch on Vulkan, readback result."""

    def __init__(self, prog, weight_prefix: str, weights_handle):
        self.prog = prog
        self.weight_prefix = weight_prefix
        self.weights_handle = weights_handle
        self.tensors: dict[str, LogicalTensor] = {}
        self._user_input_names: list[str] = []
        self._build_tensors()

    def _build_tensors(self):
        graph = self.prog.graph_module.graph
        for spec in self.prog.graph_signature.input_specs:
            for node in graph.nodes:
                if node.name == spec.arg.name:
                    tm = node.meta.get("tensor_meta")
                    if tm:
                        shape = tuple(int(d) for d in tm.shape)
                        dtype = str(tm.dtype).removeprefix("torch.")
                        if dtype in ("int64", "int32"):
                            dtype = "int32"
                        else:
                            dtype = "float32"
                        self.tensors[spec.arg.name] = LogicalTensor(
                            name=spec.arg.name,
                            spec=TensorSpec(dtype=dtype, shape=shape),
                            role=TensorRole.INPUT,
                            memory=MemoryClass.HOST_INPUT,
                            lifetime=TensorLifetime.FRAME,
                        )
                        if spec.kind not in (InputKind.PARAMETER, InputKind.BUFFER):
                            self._user_input_names.append(spec.arg.name)
                    break
        for node in graph.nodes:
            if node.op == "call_function" and node.name not in self.tensors:
                if str(node.target) in SKIP_OPS:
                    continue
                tm = node.meta.get("tensor_meta")
                if tm:
                    shape = tuple(int(d) for d in tm.shape)
                    dtype = str(tm.dtype).removeprefix("torch.")
                    if dtype in ("int64", "int32"):
                        dtype = "int32"
                    else:
                        dtype = "float32"
                    self.tensors[node.name] = LogicalTensor(
                        name=node.name,
                        spec=TensorSpec(dtype=dtype, shape=shape),
                        role=TensorRole.ACTIVATION,
                        memory=MemoryClass.FRAME_WORKSPACE,
                        lifetime=TensorLifetime.FRAME,
                    )

    @property
    def user_input_names(self) -> list[str]:
        return self._user_input_names

    def run(self, rt: RuntimeSession, frame_name: str, user_inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Register weights + inputs, dispatch all ops, readback final output — all within one frame."""
        # Register weights (load from safetensors handle on-demand)
        weight_map: dict[LogicalTensor, np.ndarray] = {}
        for spec in self.prog.graph_signature.input_specs:
            if spec.kind not in (InputKind.PARAMETER, InputKind.BUFFER):
                continue
            t = self.tensors[spec.arg.name]
            key = f"{self.weight_prefix}{spec.target}" if self.weight_prefix else spec.target
            weight_map[t] = self.weights_handle.get_tensor(key).float().numpy()
        rt.register_inputs(weight_map)

        # Register user inputs
        input_map: dict[LogicalTensor, np.ndarray] = {}
        for name, data in user_inputs.items():
            if name in self.tensors:
                input_map[self.tensors[name]] = data
        rt.register_inputs(input_map)

        # Dispatch and readback within frame
        graph = self.prog.graph_module.graph
        last_name = None
        with rt.frame(frame_name):
            for node in graph.nodes:
                if node.op != "call_function":
                    continue
                target = str(node.target)
                if target in SKIP_OPS:
                    continue
                # Skip non-float32 compute nodes (dead code from cu_seqlens mask computation)
                tm = node.meta.get("tensor_meta")
                if tm and str(tm.dtype) != "torch.float32":
                    continue
                if is_alias_op(node):
                    inputs = node_input_names(node)
                    if inputs and inputs[0] in self.tensors and node.name in self.tensors:
                        src, dst = self.tensors[inputs[0]], self.tensors[node.name]
                        rt._materialize_read(src)
                        with dst.runtime_write_scope():
                            dst.buffer = src.buffer
                            dst.descriptor_nbytes = src.descriptor_nbytes
                            dst.version = src.version
                    continue
                variant = DEFAULT_REGISTRY.resolve(node)
                if variant is None:
                    raise RuntimeError(f"No shader for {target} ({node.name})")
                inputs = node_input_names(node)
                input_fields = [f for f in variant.contract.fields if f.io_kind == IOKind.INPUT]
                output_fields = [f for f in variant.contract.fields if f.io_kind in (IOKind.OUTPUT, IOKind.INOUT)]
                kwargs = {}
                for i, field in enumerate(input_fields):
                    if i < len(inputs) and inputs[i] in self.tensors:
                        kwargs[field.name] = self.tensors[inputs[i]]
                for field in output_fields:
                    kwargs[field.name] = self.tensors[node.name]
                variant(rt, **kwargs)
                last_name = node.name
            t = self.tensors[last_name]
            return rt.device.readback_tensor(spec=t.spec, slice=t.buffer)


# ============================================================
# Audio tower
# ============================================================

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


def run_audio_tower(
    rt: RuntimeSession,
    model,
    config,
    weights,
    input_features: np.ndarray,
    feature_lens: np.ndarray,
) -> np.ndarray:
    """Audio tower: CPU control flow + export-based Vulkan compute for all ops."""
    ac = config.thinker_config.audio_config
    n_window = 50
    n_window_infer = 800

    # CPU: chunking and padding
    feat_len = int(feature_lens[0])
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)

    chunk_num = int(np.ceil(feat_len / (n_window * 2)))
    chunk_lengths = np.full(chunk_num, n_window * 2, dtype=np.int64)
    remainder = feat_len % (n_window * 2)
    if remainder != 0:
        chunk_lengths[-1] = remainder

    features_t = input_features.T  # (feat_len, num_mel_bins)
    chunks = []
    offset = 0
    for cl in chunk_lengths:
        chunks.append(features_t[offset:offset + cl])
        offset += cl

    max_chunk_len = int(chunk_lengths.max())
    num_chunks = len(chunks)
    num_mel = input_features.shape[0]
    padded_feature = np.zeros((num_chunks, 1, num_mel, max_chunk_len), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        padded_feature[i, 0, :, :chunk.shape[0]] = chunk.T

    at = model.thinker.audio_tower

    # Conv2d1 + GELU
    print(f"  conv2d1 ({padded_feature.shape})...")
    conv1_prog = export_submodule(at.conv2d1.float(), args=(torch.zeros(padded_feature.shape, device="meta"),))
    conv1_mod = ExportedSubmodule(conv1_prog, "thinker.audio_tower.conv2d1.", weights)
    conv1_out = _gelu_numpy(conv1_mod.run(rt, "spike.audio.conv2d1", {conv1_mod.user_input_names[0]: padded_feature}))

    # Conv2d2 + GELU
    print(f"  conv2d2 ({conv1_out.shape})...")
    conv2_prog = export_submodule(at.conv2d2.float(), args=(torch.zeros(conv1_out.shape, device="meta"),))
    conv2_mod = ExportedSubmodule(conv2_prog, "thinker.audio_tower.conv2d2.", weights)
    conv2_out = _gelu_numpy(conv2_mod.run(rt, "spike.audio.conv2d2", {conv2_mod.user_input_names[0]: conv1_out}))

    # Conv2d3 + GELU
    print(f"  conv2d3 ({conv2_out.shape})...")
    conv3_prog = export_submodule(at.conv2d3.float(), args=(torch.zeros(conv2_out.shape, device="meta"),))
    conv3_mod = ExportedSubmodule(conv3_prog, "thinker.audio_tower.conv2d3.", weights)
    conv3_out = _gelu_numpy(conv3_mod.run(rt, "spike.audio.conv2d3", {conv3_mod.user_input_names[0]: conv2_out}))

    # CPU: reshape (b, c, f, t) -> (b, t, c*f)
    b, c, f, t = conv3_out.shape
    reshaped = conv3_out.transpose(0, 3, 1, 2).reshape(b, t, c * f)

    # conv_out (Linear no bias): flatten to (b*t, 7680) -> (b*t, 896)
    conv_out_input = reshaped.reshape(b * t, c * f)
    print(f"  conv_out ({conv_out_input.shape})...")
    conv_out_prog = export_submodule(at.conv_out.float(), args=(torch.zeros(conv_out_input.shape, device="meta"),))
    conv_out_mod = ExportedSubmodule(conv_out_prog, "thinker.audio_tower.conv_out.", weights)
    conv_out_result = conv_out_mod.run(rt, "spike.audio.conv_out", {conv_out_mod.user_input_names[0]: conv_out_input})
    conv_out_result = conv_out_result.reshape(b, t, ac.d_model)

    # CPU: add positional embedding
    pos_emb = _compute_positional_embedding(t, ac.d_model)
    padded_embed = conv_out_result + pos_emb[None, :t, :]

    # CPU: compact (select valid positions using mask)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    valid_positions = []
    for i, fl in enumerate(feature_lens_after_cnn):
        for j in range(int(fl)):
            valid_positions.append((i, j))
    hidden_states = np.array([padded_embed[i, j] for i, j in valid_positions], dtype=np.float32)
    print(f"  hidden_states after compact: {hidden_states.shape}")

    # CPU: compute cu_seqlens
    window_aftercnn = int(feature_lens_after_cnn.max()) * (n_window_infer // (n_window * 2))
    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens:
        cnn_len = int(cnn_len)
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        rem = cnn_len % window_aftercnn
        if rem != 0:
            cu_chunk_lens += [rem]
    cu_seqlens = np.cumsum(cu_chunk_lens, dtype=np.int32)

    # Encoder layers × 18
    seq_len = hidden_states.shape[0]
    print(f"  Exporting encoder layer (seq_len={seq_len})...")
    enc_prog = export_submodule(
        at.layers[0].float(),
        args=(torch.zeros(seq_len, ac.d_model, device="meta"), torch.zeros(len(cu_seqlens), dtype=torch.int32, device="meta")),
    )

    for layer_idx in range(ac.encoder_layers):
        enc_mod = ExportedSubmodule(enc_prog, f"thinker.audio_tower.layers.{layer_idx}.", weights)
        enc_inputs = {}
        for name in enc_mod.user_input_names:
            t_spec = enc_mod.tensors[name].spec
            if t_spec.dtype == "int32":
                enc_inputs[name] = cu_seqlens
            else:
                enc_inputs[name] = hidden_states
        hidden_states = enc_mod.run(rt, f"spike.audio.enc.{layer_idx}", enc_inputs)
        if layer_idx == 0:
            print(f"    layer 0: max={hidden_states.max():.4f}")
        if layer_idx % 6 == 5:
            print(f"    layer {layer_idx} done")

    # ln_post
    print("  ln_post...")
    ln_prog = export_submodule(at.ln_post.float(), args=(torch.zeros(seq_len, ac.d_model, device="meta"),))
    ln_mod = ExportedSubmodule(ln_prog, "thinker.audio_tower.ln_post.", weights)
    hidden_states = ln_mod.run(rt, "spike.audio.ln_post", {ln_mod.user_input_names[0]: hidden_states})

    # proj1 + gelu
    print("  proj1 + gelu...")
    proj1_prog = export_submodule(at.proj1.float(), args=(torch.zeros(seq_len, 896, device="meta"),))
    proj1_mod = ExportedSubmodule(proj1_prog, "thinker.audio_tower.proj1.", weights)
    hidden_states = _gelu_numpy(proj1_mod.run(rt, "spike.audio.proj1", {proj1_mod.user_input_names[0]: hidden_states}))

    # proj2
    print("  proj2...")
    proj2_prog = export_submodule(at.proj2.float(), args=(torch.zeros(seq_len, 896, device="meta"),))
    proj2_mod = ExportedSubmodule(proj2_prog, "thinker.audio_tower.proj2.", weights)
    audio_hidden = proj2_mod.run(rt, "spike.audio.proj2", {proj2_mod.user_input_names[0]: hidden_states})
    print(f"  Audio tower output: {audio_hidden.shape}")

    return audio_hidden


# ============================================================
# Text pipeline
# ============================================================

def _compute_rope(seq_len: int, head_dim: int, rope_theta: float = 5_000_000.0, start_pos: int = 0) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(start_pos, start_pos + seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32).reshape(1, seq_len, head_dim)
    sin = np.sin(emb).astype(np.float32).reshape(1, seq_len, head_dim)
    return cos, sin


def run_text_prefill(
    rt: RuntimeSession,
    model,
    config,
    weights,
    input_ids: np.ndarray,
    audio_hidden: np.ndarray,
) -> int:
    """Text prefill: embed → inject audio → decoder layers → norm → lm_head → argmax. Returns first token."""
    tc = config.thinker_config.text_config
    prompt_length = input_ids.shape[1]

    # Embedding
    print("  embed_tokens...")
    embed = model.thinker.model.embed_tokens.float()
    embed_prog = export_submodule(embed, args=(torch.zeros((1, prompt_length), dtype=torch.long, device="meta"),))
    embed_mod = ExportedSubmodule(embed_prog, "thinker.model.embed_tokens.", weights)
    embedded = embed_mod.run(rt, "spike.text.embed", {embed_mod.user_input_names[0]: input_ids.astype(np.int32)})
    print(f"    shape: {embedded.shape}")

    # Inject audio hidden states at placeholder positions
    ids_flat = input_ids.flatten()
    audio_token_id = 151646
    audio_positions = np.where(ids_flat == audio_token_id)[0]
    if len(audio_positions) > 0:
        audio_start = int(audio_positions[0])
        audio_end = audio_start + audio_hidden.shape[0]
        print(f"    Injecting audio [{audio_start}:{audio_end}]")
        embedded[0, audio_start:audio_end, :] = audio_hidden

    # RoPE
    rope_cos, rope_sin = _compute_rope(prompt_length, tc.head_dim)

    # Decoder layers × N
    print(f"  Decoder layers × {tc.num_hidden_layers}...")
    layer = model.thinker.model.layers[0].float()
    layer_prog = export_submodule(
        layer,
        args=(torch.zeros(1, prompt_length, tc.hidden_size, device="meta"),),
        kwargs={"position_embeddings": (
            torch.zeros(1, prompt_length, tc.head_dim, device="meta"),
            torch.zeros(1, prompt_length, tc.head_dim, device="meta"),
        )},
    )

    hidden_states = embedded
    for layer_idx in range(tc.num_hidden_layers):
        layer_mod = ExportedSubmodule(layer_prog, f"thinker.model.layers.{layer_idx}.", weights)
        user_inputs = {}
        for name in layer_mod.user_input_names:
            if "hidden" in name:
                user_inputs[name] = hidden_states
            elif "position_embeddings_0" in name:
                user_inputs[name] = rope_cos
            elif "position_embeddings_1" in name:
                user_inputs[name] = rope_sin
        hidden_states = layer_mod.run(rt, f"spike.text.layer.{layer_idx}", user_inputs)
        if layer_idx == 0:
            print(f"    layer 0: max={hidden_states.max():.4f}")
        if layer_idx % 7 == 6:
            print(f"    layer {layer_idx} done")

    print(f"  All layers done.")

    # Final norm
    print("  final_norm...")
    norm = model.thinker.model.norm.float()
    norm_prog = export_submodule(norm, args=(torch.zeros(1, prompt_length, tc.hidden_size, device="meta"),))
    norm_mod = ExportedSubmodule(norm_prog, "thinker.model.norm.", weights)
    normed = norm_mod.run(rt, "spike.text.norm", {norm_mod.user_input_names[0]: hidden_states})

    # LM head (last token only)
    print("  lm_head...")
    last_hidden = normed[:, -1:, :]
    lm_head = model.thinker.lm_head.float()
    lm_prog = export_submodule(lm_head, args=(torch.zeros(1, 1, tc.hidden_size, device="meta"),))
    lm_mod = ExportedSubmodule(lm_prog, "thinker.lm_head.", weights)
    logits = lm_mod.run(rt, "spike.text.lm_head", {lm_mod.user_input_names[0]: last_hidden})

    first_token = int(np.argmax(logits[0, -1, :]))
    print(f"  First token: {first_token}")
    return first_token


def run_decode_step(
    rt: RuntimeSession,
    model,
    config,
    weights,
    token_id: int,
    cache_position: int,
    decode_layer_prog,
    decode_embed_mod: ExportedSubmodule,
    decode_norm_mod: ExportedSubmodule,
    decode_lm_mod: ExportedSubmodule,
) -> int:
    """One decode step: embed → layers → norm → lm_head → argmax. Returns next token."""
    tc = config.thinker_config.text_config

    # Embed
    hidden = decode_embed_mod.run(
        rt, "spike.decode.embed",
        {decode_embed_mod.user_input_names[0]: np.array([[token_id]], dtype=np.int32)},
    )

    # RoPE
    rope_cos, rope_sin = _compute_rope(1, tc.head_dim, start_pos=cache_position)

    # Decoder layers
    for layer_idx in range(tc.num_hidden_layers):
        layer_mod = ExportedSubmodule(decode_layer_prog, f"thinker.model.layers.{layer_idx}.", weights)
        user_inputs = {}
        for name in layer_mod.user_input_names:
            if "hidden" in name:
                user_inputs[name] = hidden
            elif "position_embeddings_0" in name:
                user_inputs[name] = rope_cos
            elif "position_embeddings_1" in name:
                user_inputs[name] = rope_sin
        hidden = layer_mod.run(rt, f"spike.decode.layer.{layer_idx}", user_inputs)

    # Norm + LM head
    normed = decode_norm_mod.run(rt, "spike.decode.norm", {decode_norm_mod.user_input_names[0]: hidden})
    logits = decode_lm_mod.run(rt, "spike.decode.lm_head", {decode_lm_mod.user_input_names[0]: normed})
    return int(np.argmax(logits[0, -1, :]))


# ============================================================
# Main
# ============================================================

def main() -> int:
    wav_path = Path("tests/fixtures/qwen3_asr_asknot.wav")
    if not wav_path.exists():
        print(f"ERROR: Test wav not found at {wav_path}")
        return 1

    print("Loading model on meta device...")
    model, config, model_dir = _load_model_and_config()
    tc = config.thinker_config.text_config

    print("Opening weights (memory-mapped)...")
    weights = safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu")

    print("Preparing audio inputs (CPU preprocessing)...")
    processor, prepared = prepare_qwen3_asr_inputs(model_dir=model_dir, wav=str(wav_path))
    print(f"  prompt_length={prepared.prompt_length}, audio_feature_length={prepared.audio_feature_length}")

    rt = RuntimeSession.open(device_index=0, model_dir=model_dir)

    # Phase 1: Audio Tower
    print("\n=== Phase 1: Audio Tower ===")
    audio_feature_len = prepared.audio_feature_length
    input_features = np.ascontiguousarray(
        prepared.input_features[0, :, :audio_feature_len],
        dtype=np.float32,
    )
    feature_lens = np.array([audio_feature_len], dtype=np.int64)
    audio_hidden = run_audio_tower(rt, model, config, weights, input_features, feature_lens)

    gc.collect()

    # Phase 2: Text Prefill
    print("\n=== Phase 2: Text Prefill ===")
    first_token = run_text_prefill(rt, model, config, weights, prepared.input_ids, audio_hidden)

    gc.collect()

    # Phase 3: Decode Loop
    print("\n=== Phase 3: Decode Loop ===")
    max_new_tokens = 20
    eos_token_ids = {151645, 151643}
    generated_tokens = [first_token]

    # Pre-export decode-step submodules (seq_len=1)
    decode_layer_prog = export_submodule(
        model.thinker.model.layers[0].float(),
        args=(torch.zeros(1, 1, tc.hidden_size, device="meta"),),
        kwargs={"position_embeddings": (
            torch.zeros(1, 1, tc.head_dim, device="meta"),
            torch.zeros(1, 1, tc.head_dim, device="meta"),
        )},
    )
    decode_embed_prog = export_submodule(
        model.thinker.model.embed_tokens.float(),
        args=(torch.zeros((1, 1), dtype=torch.long, device="meta"),),
    )
    decode_embed_mod = ExportedSubmodule(decode_embed_prog, "thinker.model.embed_tokens.", weights)

    decode_norm_prog = export_submodule(
        model.thinker.model.norm.float(),
        args=(torch.zeros(1, 1, tc.hidden_size, device="meta"),),
    )
    decode_norm_mod = ExportedSubmodule(decode_norm_prog, "thinker.model.norm.", weights)

    decode_lm_prog = export_submodule(
        model.thinker.lm_head.float(),
        args=(torch.zeros(1, 1, tc.hidden_size, device="meta"),),
    )
    decode_lm_mod = ExportedSubmodule(decode_lm_prog, "thinker.lm_head.", weights)

    for step in range(max_new_tokens - 1):
        if generated_tokens[-1] in eos_token_ids:
            print(f"  EOS at step {step}")
            break

        cache_pos = prepared.prompt_length + step
        next_token = run_decode_step(
            rt, model, config, weights, generated_tokens[-1], cache_pos,
            decode_layer_prog, decode_embed_mod, decode_norm_mod, decode_lm_mod,
        )
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
