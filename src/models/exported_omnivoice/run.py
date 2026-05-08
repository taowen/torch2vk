"""OmniVoice TTS inference using exported Vulkan shaders.

32-step iterative masked decoding with classifier-free guidance.
Embedding computed host-side (PyTorch); LLM + audio_head on Vulkan GPU.

Run from project root:
    .venv/bin/python -m models.exported_omnivoice.run
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.hf_cache import resolve_cached_model
from models.exported_omnivoice import dispatch
from models.exported_omnivoice.tensors.audio_head import (
    AUDIO_HEAD_OUTPUT,
    AudioHeadTensors,
    create_audio_head,
)
from models.exported_omnivoice.tensors.llm_forward import (
    LLM_FORWARD_OUTPUT,
    LlmForwardTensors,
    create_llm_forward,
)
from models.optimized_omnivoice.pytorch.example import REPO_ID, SAMPLE_RATE, save_audio_wav
from omnivoice import OmniVoice, OmniVoiceConfig
from torch2vk.runtime.logical import LogicalTensor, TensorRole
from torch2vk.runtime.session import RuntimeSession

DEFAULT_TEXT = "hello world this is a speech recognition test"
DEFAULT_OUTPUT_WAV = Path("/tmp/torch2vk_omnivoice_exported.wav")


def _ensure_bfloat16_checkpoint(model_dir: Path) -> Path:
    """Convert float32 safetensors to bfloat16 (cached in a subdirectory)."""
    bf16_dir = model_dir / "bf16"
    dst = bf16_dir / "model.safetensors"
    if dst.exists():
        return bf16_dir
    bf16_dir.mkdir(exist_ok=True)
    from safetensors.torch import load_file, save_file
    state_dict = load_file(str(model_dir / "model.safetensors"))
    bf16_dict = {}
    for k, v in state_dict.items():
        if v.dtype == torch.float32:
            bf16_dict[k] = v.to(torch.bfloat16)
        else:
            bf16_dict[k] = v
    save_file(bf16_dict, str(dst))
    return bf16_dir


def _materialize_weights(rt: RuntimeSession, tensors_obj: object) -> None:
    for f in dataclasses.fields(tensors_obj):
        value = getattr(tensors_obj, f.name)
        if isinstance(value, list):
            for item in value:
                _materialize_weights(rt, item)
            continue
        if not isinstance(value, LogicalTensor):
            continue
        if value.role is not TensorRole.WEIGHT:
            continue
        if value.buffer is not None:
            continue
        rt._materialize_weight(value)


def _compute_rope(seq_len: int, head_dim: int, batch: int = 2, theta: float = 1_000_000.0) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = np.outer(positions, freqs)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    cos_full = np.concatenate([cos, cos], axis=-1)[None, :, :]
    sin_full = np.concatenate([sin, sin], axis=-1)[None, :, :]
    cos_full = np.broadcast_to(cos_full, (batch, seq_len, head_dim)).copy()
    sin_full = np.broadcast_to(sin_full, (batch, seq_len, head_dim)).copy()
    return cos_full, sin_full


def _get_time_steps(t_start: float, t_end: float, num_step: int, t_shift: float) -> np.ndarray:
    t = np.linspace(t_start, t_end, num_step + 1, dtype=np.float64)
    t = t / (t + t_shift - t_shift * t)
    return t.astype(np.float32)


def _predict_tokens_with_scoring(
    c_logits: torch.Tensor,
    u_logits: torch.Tensor,
    guidance_scale: float,
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if guidance_scale != 0:
        c_log_probs = F.log_softmax(c_logits, dim=-1)
        u_log_probs = F.log_softmax(u_logits, dim=-1)
        log_probs = torch.log_softmax(
            c_log_probs + guidance_scale * (c_log_probs - u_log_probs),
            dim=-1,
        )
    else:
        log_probs = F.log_softmax(c_logits, dim=-1)

    log_probs[..., mask_id] = -float("inf")
    pred_tokens = log_probs.argmax(dim=-1)
    confidence_scores = log_probs.max(dim=-1)[0]
    return pred_tokens, confidence_scores


def main(
    *,
    text: str = DEFAULT_TEXT,
    output: str | Path = DEFAULT_OUTPUT_WAV,
    pytorch_compare: bool = False,
    num_steps: int = 32,
) -> Path:
    output_path = Path(output)
    model_dir = resolve_cached_model(REPO_ID)
    config_data = json.loads((model_dir / "config.json").read_text())
    config = OmniVoiceConfig(**config_data)

    print("Loading embedding weights...")
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    text_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    state_dict = load_file(str(model_dir / "model.safetensors"), device="cuda")

    embed_tokens_weight = state_dict["llm.embed_tokens.weight"].float()
    audio_embed_weight = state_dict["audio_embeddings.weight"].float()
    codebook_offsets = torch.arange(config.num_audio_codebook, device="cuda") * config.audio_vocab_size
    del state_dict
    torch.cuda.empty_cache()
    print(f"  embed_tokens: {embed_tokens_weight.shape}, audio_embed: {audio_embed_weight.shape}")

    llm_config = config.llm_config
    num_audio_codebook = config.num_audio_codebook
    audio_vocab_size = config.audio_vocab_size
    audio_mask_id = config.audio_mask_id
    hidden_size = llm_config.hidden_size
    head_dim = llm_config.head_dim
    num_layers = llm_config.num_hidden_layers

    # Exported model uses fixed seq_len=300
    EXPORTED_SEQ_LEN = 300

    # Prepare inputs (host-side _prepare_inference_inputs)
    print("Preparing inputs...")
    style_text = "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_tokens = text_tokenizer(style_text, return_tensors="pt").input_ids
    style_tokens = style_tokens.repeat(num_audio_codebook, 1).unsqueeze(0).cuda()

    wrapped_text = f"<|text_start|>{text}<|text_end|>"
    text_tokens = text_tokenizer(wrapped_text, return_tensors="pt").input_ids
    text_tokens = text_tokens.repeat(num_audio_codebook, 1).unsqueeze(0).cuda()

    prefix_len = style_tokens.shape[2] + text_tokens.shape[2]
    target_len = EXPORTED_SEQ_LEN - prefix_len
    if target_len <= 0:
        raise ValueError(f"Text too long: prefix={prefix_len} exceeds EXPORTED_SEQ_LEN={EXPORTED_SEQ_LEN}")

    target_audio_tokens = torch.full(
        (1, num_audio_codebook, target_len),
        audio_mask_id,
        dtype=torch.long,
        device="cuda",
    )

    cond_input_ids = torch.cat([style_tokens, text_tokens, target_audio_tokens], dim=2)
    cond_total_len = cond_input_ids.shape[2]
    assert cond_total_len == EXPORTED_SEQ_LEN
    cond_audio_start = cond_total_len - target_len

    cond_audio_mask = torch.zeros(1, cond_total_len, dtype=torch.bool, device="cuda")
    cond_audio_mask[0, cond_audio_start:] = True

    # Build CFG batch (2 = cond + uncond), padded to EXPORTED_SEQ_LEN
    B = 1
    seq_len = EXPORTED_SEQ_LEN
    c_len = cond_total_len
    u_len = target_len

    batch_input_ids = torch.full(
        (2 * B, num_audio_codebook, seq_len),
        audio_mask_id,
        dtype=torch.long,
        device="cuda",
    )
    batch_audio_mask = torch.zeros(2 * B, seq_len, dtype=torch.bool, device="cuda")
    batch_attention_mask = torch.zeros(2 * B, 1, seq_len, seq_len, dtype=torch.bool, device="cuda")

    # Cond
    batch_input_ids[0, :, :c_len] = cond_input_ids[0]
    batch_audio_mask[0, :c_len] = cond_audio_mask[0]
    batch_attention_mask[0, :, :c_len, :c_len] = True

    # Uncond
    batch_input_ids[B, :, :u_len] = cond_input_ids[0, :, -u_len:]
    batch_audio_mask[B, :u_len] = cond_audio_mask[0, -u_len:]
    batch_attention_mask[B, :, :u_len, :u_len] = True
    if seq_len > u_len:
        pad_diag = torch.arange(u_len, seq_len, device="cuda")
        batch_attention_mask[B, :, pad_diag, pad_diag] = True

    # Convert attention mask to additive float
    attn_mask_float = torch.zeros_like(batch_attention_mask, dtype=torch.float32)
    attn_mask_float.masked_fill_(~batch_attention_mask, -torch.finfo(torch.float32).max)

    print(f"  seq_len={seq_len}, target_len={target_len}, cond_audio_start={cond_audio_start}")

    # Pre-compute RoPE (replicate to batch=2 for CFG)
    cos_np, sin_np = _compute_rope(seq_len, head_dim)
    attn_mask_np = attn_mask_float.cpu().numpy()

    # Create runtime and tensors
    print("Initializing Vulkan runtime...")
    bf16_dir = _ensure_bfloat16_checkpoint(model_dir)
    rt = RuntimeSession.open(device_index=0, model_dir=bf16_dir)

    llm_t = create_llm_forward(
        "omnivoice.llm",
        request_state_outputs={LLM_FORWARD_OUTPUT},
    )
    audio_head_t = create_audio_head(
        "omnivoice.audio_head",
        bindings={"input": llm_t.mul_365},
        request_state_outputs={AUDIO_HEAD_OUTPUT},
    )

    print("Materializing weights...")
    _materialize_weights(rt, llm_t)
    _materialize_weights(rt, audio_head_t)
    stats = rt.device.allocation_stats()
    vram_after_weights = torch.cuda.memory_allocated() / 1024**2
    print(f"  Vulkan weights: {stats.device_local_live_bytes / 1024**2:.1f} MB")
    print(f"  PyTorch CUDA (embeddings): {vram_after_weights:.0f} MB")
    print(f"  Total VRAM: {(stats.device_local_live_bytes / 1024**2 + vram_after_weights):.0f} MB")

    # Iterative decoding
    print(f"\n=== Iterative Decoding ({num_steps} steps) ===")
    tokens = torch.full(
        (B, num_audio_codebook, target_len),
        audio_mask_id,
        dtype=torch.long,
        device="cuda",
    )

    timesteps = _get_time_steps(0.0, 1.0, num_steps, t_shift=0.1)
    total_mask = target_len * num_audio_codebook
    schedule = []
    rem = total_mask
    for step in range(num_steps):
        if step == num_steps - 1:
            num = rem
        else:
            num = min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
        schedule.append(int(num))
        rem -= int(num)

    layer_ids = torch.arange(num_audio_codebook, device="cuda").view(1, -1, 1)
    guidance_scale = 2.0
    layer_penalty_factor = 5.0
    position_temperature = 5.0

    for step in range(num_steps):
        k = schedule[step]
        if k <= 0:
            continue

        # Embedding on CUDA (only embed weights, not full model)
        with torch.no_grad():
            text_embeds = embed_tokens_weight[batch_input_ids[:, 0, :]]
            shifted_ids = (
                batch_input_ids * batch_audio_mask.unsqueeze(1).long()
            ) + codebook_offsets.view(1, -1, 1)
            audio_embeds = audio_embed_weight[shifted_ids].sum(dim=1)
            hidden_states = torch.where(
                batch_audio_mask.unsqueeze(-1), audio_embeds, text_embeds
            )
        hidden_np = hidden_states.cpu().numpy()

        # GPU frame: LLM forward + audio head
        rt._inputs.clear()
        rt.register_inputs({
            llm_t.hidden_states: hidden_np,
            llm_t.cos: cos_np,
            llm_t.sin: sin_np,
            llm_t.attention_mask: attn_mask_np,
        })

        frame_scope = rt.frame(f"omnivoice.step.{step}")

        with frame_scope:
            dispatch.run_llm_forward(rt, llm_t)
            dispatch.run_audio_head(rt, audio_head_t)

        # Read back logits
        logits_raw = rt.read_request_state(audio_head_t.linear)
        logits_t = torch.from_numpy(logits_raw).cuda()

        # Reshape: (2, S, C*V) → (2, C, S, V)
        batch_logits = logits_t.view(
            2, seq_len, num_audio_codebook, audio_vocab_size
        ).permute(0, 2, 1, 3).float()

        # Extract target logits
        c_logits = batch_logits[0:1, :, c_len - target_len:c_len, :]
        u_logits = batch_logits[1:2, :, :target_len, :]

        # CFG + scoring
        pred_tokens, scores = _predict_tokens_with_scoring(
            c_logits, u_logits, guidance_scale, audio_mask_id,
        )

        # Layer penalty + gumbel noise
        scores = scores - (layer_ids * layer_penalty_factor)
        if position_temperature > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
            scores = scores + gumbel * position_temperature

        # Mask out already-unmasked tokens
        sample_tokens = tokens[0:1, :, :target_len]
        scores.masked_fill_(sample_tokens != audio_mask_id, -float("inf"))

        # Top-k unmask
        _, topk_idx = torch.topk(scores.flatten(), k)
        flat_tokens = sample_tokens.flatten().clone()
        flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
        tokens[0] = flat_tokens.view(num_audio_codebook, target_len)

        # Update batch structures
        batch_input_ids[0, :, c_len - target_len:c_len] = tokens[0]
        batch_input_ids[B, :, :target_len] = tokens[0]

        if step % 8 == 0 or step == num_steps - 1:
            unmasked = (tokens[0] != audio_mask_id).sum().item()
            total = num_audio_codebook * target_len
            print(f"  Step {step}: unmasked {unmasked}/{total} ({100*unmasked/total:.0f}%)")

    # Decode audio tokens
    print("\nDecoding audio tokens...")
    from transformers.models.higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerModel
    audio_tokenizer_path = model_dir / "audio_tokenizer"
    audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
        str(audio_tokenizer_path), device_map="cuda"
    )
    audio_output = audio_tokenizer.decode(tokens[0:1].cuda())
    waveform = audio_output.audio_values[0].cpu()

    # Save wav
    output_path = save_audio_wav(waveform, output_path)
    print(f"\nOutput: {output_path}")
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
