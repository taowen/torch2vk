"""OmniVoice TTS inference using exported Vulkan shaders.

32-step iterative masked decoding with classifier-free guidance.
Embedding, LLM, audio_head, CFG scoring, and token updates run on Vulkan.

Run from project root:
    .venv/bin/python -m models.exported_omnivoice.run
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol, Sequence, cast

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoTokenizer
from transformers.models.higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerModel

from models.hf_cache import resolve_cached_model
from models.exported_omnivoice import dispatch, reference, reference_setup
from models.exported_omnivoice.shaders import model_shaders
from models.exported_omnivoice.tensors.model import create_model_tensors, model_tensors
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
from models.optimized_omnivoice.pytorch.example import REPO_ID, save_audio_wav
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.replay import ReplayPlan, execute_replay, stage_replay_step_inputs
from torch2vk.runtime.session import RuntimeSession

DEFAULT_TEXT = "hello world this is a speech recognition test"
DEFAULT_OUTPUT_WAV = Path("/tmp/torch2vk_omnivoice_exported.wav")


class _AudioDecodeOutput(Protocol):
    audio_values: Sequence[torch.Tensor]


def _ensure_bfloat16_checkpoint(model_dir: Path) -> Path:
    """Convert float32 safetensors to bfloat16 (cached in a subdirectory)."""
    bf16_dir = model_dir / "bf16"
    dst = bf16_dir / "model.safetensors"
    if dst.exists():
        return bf16_dir
    bf16_dir.mkdir(exist_ok=True)
    state_dict = load_file(str(model_dir / "model.safetensors"))
    bf16_dict = {}
    for k, v in state_dict.items():
        if v.dtype == torch.float32:
            bf16_dict[k] = v.to(torch.bfloat16)
        else:
            bf16_dict[k] = v
    save_file(bf16_dict, str(dst))
    return bf16_dir


def _round_model_float_weights_to_bfloat16(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(torch.bfloat16).to(torch.float32)
    for buffer in model.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(torch.bfloat16).to(torch.float32)


def _get_time_steps(t_start: float, t_end: float, num_step: int, t_shift: float) -> np.ndarray:
    t = np.linspace(t_start, t_end, num_step + 1, dtype=np.float64)
    t = t / (t + t_shift - t_shift * t)
    return t.astype(np.float32)


def _expected_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.from_numpy(np.ascontiguousarray(as_numpy_array(value))).cuda()


def _run_generation_step_with_compare(
    rt: RuntimeSession,
    *,
    step: int,
    unmask_count: int,
    refs: reference_setup.CompareReferences,
) -> None:
    with rt.frame(f"omnivoice.step.{step:04d}"):
        dispatch.run_input_embed(rt)
        input_embed_expected = reference.run_input_embed(
            rt,
            refs.input_embed,
            step=step,
            input_ids=refs.batch_input_ids,
            audio_mask=refs.batch_audio_mask,
        )
        hidden_states = _expected_tensor(input_embed_expected["hidden_states"]).float()

        dispatch.run_llm_forward(rt)
        llm_expected = reference.run_llm_forward(
            rt,
            refs.llm_forward,
            step=step,
            hidden_states=hidden_states,
            cos=refs.rope_cos,
            sin=refs.rope_sin,
            attention_mask=refs.attention_mask,
        )
        llm_output = _expected_tensor(llm_expected["mul_365"]).float()

        dispatch.run_audio_head(rt)
        audio_head_expected = reference.run_audio_head(
            rt,
            refs.audio_head,
            step=step,
            input=llm_output,
        )
        logits = _expected_tensor(audio_head_expected["linear"]).float()

        step_index = torch.tensor([step], dtype=torch.int64, device="cuda")
        dispatch.run_token_score(rt)
        token_score_expected = reference.run_token_score(
            rt,
            refs.token_score,
            step=step,
            logits=logits,
            tokens=refs.tokens,
            audio_mask_id=refs.audio_mask_id,
            rng_seed=refs.rng_seed,
            step_index=step_index,
        )
        candidate_tokens = _expected_tensor(token_score_expected["candidate_tokens"]).long()
        candidate_scores = _expected_tensor(token_score_expected["candidate_scores"]).float()

        dispatch.run_token_update(rt)
        unmask_count_t = torch.tensor([unmask_count], dtype=torch.uint32, device="cuda")
        token_update_expected = reference.run_token_update(
            rt,
            refs.token_update,
            step=step,
            tokens=refs.tokens,
            batch_input_ids=refs.batch_input_ids,
            candidate_tokens=candidate_tokens,
            candidate_scores=candidate_scores,
            unmask_count=unmask_count_t,
        )
        refs.tokens = _expected_tensor(token_update_expected["tokens"]).long()
        refs.batch_input_ids = _expected_tensor(token_update_expected["batch_input_ids"]).long()


def _generation_step_inputs(step: int, unmask_count: int) -> dict[LogicalTensor, np.ndarray]:
    return {
        model_tensors().step_index: np.array([step], dtype=np.uint32),
        model_tensors().unmask_count: np.array([unmask_count], dtype=np.uint32),
    }


def _build_generation_replay_plan(
    rt: RuntimeSession,
    *,
    frame: str,
) -> ReplayPlan:
    plan = rt.build_replay_plan(
        name="exported_omnivoice_generation_step",
        frame=frame,
    )
    if plan.readback_slots:
        raise RuntimeError("OmniVoice generation replay must not use readback slots")
    return plan


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

    print("Loading tokenizer...")
    text_tokenizer = AutoTokenizer.from_pretrained(model_dir)

    num_audio_codebook = config.num_audio_codebook
    audio_mask_id = config.audio_mask_id

    # Exported model uses fixed seq_len=300
    EXPORTED_SEQ_LEN = 300

    # Prepare inputs (host-side _prepare_inference_inputs)
    print("Preparing inputs...")
    style_text = "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = text_tokenizer(style_text, return_tensors="pt").input_ids.numpy().astype(np.int64)
    style_tokens = np.broadcast_to(
        style_ids,
        (num_audio_codebook, style_ids.shape[1]),
    )[None, :, :].copy()

    wrapped_text = f"<|text_start|>{text}<|text_end|>"
    text_ids = text_tokenizer(wrapped_text, return_tensors="pt").input_ids.numpy().astype(np.int64)
    text_tokens = np.broadcast_to(
        text_ids,
        (num_audio_codebook, text_ids.shape[1]),
    )[None, :, :].copy()

    prefix_len = style_tokens.shape[2] + text_tokens.shape[2]
    target_len = EXPORTED_SEQ_LEN - prefix_len
    if target_len <= 0:
        raise ValueError(f"Text too long: prefix={prefix_len} exceeds EXPORTED_SEQ_LEN={EXPORTED_SEQ_LEN}")

    target_audio_tokens = np.full(
        (1, num_audio_codebook, target_len),
        audio_mask_id,
        dtype=np.int64,
    )

    cond_input_ids = np.concatenate([style_tokens, text_tokens, target_audio_tokens], axis=2)
    cond_total_len = cond_input_ids.shape[2]
    assert cond_total_len == EXPORTED_SEQ_LEN
    cond_audio_start = cond_total_len - target_len

    cond_audio_mask = np.zeros((1, cond_total_len), dtype=np.uint32)
    cond_audio_mask[0, cond_audio_start:] = 1

    # Build CFG batch (2 = cond + uncond), padded to EXPORTED_SEQ_LEN
    B = 1
    seq_len = EXPORTED_SEQ_LEN
    c_len = cond_total_len
    u_len = target_len

    batch_input_ids = np.full(
        (2 * B, num_audio_codebook, seq_len),
        audio_mask_id,
        dtype=np.int64,
    )
    batch_audio_mask = np.zeros((2 * B, seq_len), dtype=np.uint32)
    batch_attention_mask = np.zeros((2 * B, 1, seq_len, seq_len), dtype=np.bool_)

    # Cond
    batch_input_ids[0, :, :c_len] = cond_input_ids[0]
    batch_audio_mask[0, :c_len] = cond_audio_mask[0]
    batch_attention_mask[0, :, :c_len, :c_len] = True

    # Uncond
    batch_input_ids[B, :, :u_len] = cond_input_ids[0, :, -u_len:]
    batch_audio_mask[B, :u_len] = cond_audio_mask[0, -u_len:]
    batch_attention_mask[B, :, :u_len, :u_len] = True
    if seq_len > u_len:
        pad_diag = np.arange(u_len, seq_len)
        batch_attention_mask[B, :, pad_diag, pad_diag] = True

    # Convert attention mask to additive float
    attn_mask_np = np.zeros(batch_attention_mask.shape, dtype=np.float32)
    attn_mask_np[~batch_attention_mask] = -np.finfo(np.float32).max

    print(f"  seq_len={seq_len}, target_len={target_len}, cond_audio_start={cond_audio_start}")

    # Create runtime and tensors
    print("Initializing Vulkan runtime...")
    bf16_dir = _ensure_bfloat16_checkpoint(model_dir)

    print("Declaring tensors...")
    create_model_tensors(target_len=target_len)
    rt = RuntimeSession.open(
        device_index=0,
        model_dir=bf16_dir,
        model_tensors=model_tensors(),
        model_shaders=model_shaders(),
    )

    # Iterative decoding
    print(f"\n=== Iterative Decoding ({num_steps} steps) ===")
    tokens = np.full(
        (B, num_audio_codebook, target_len),
        audio_mask_id,
        dtype=np.int64,
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

    compare_refs = None
    rng_seed = 0x1234ABCD
    if pytorch_compare:
        print("Loading PyTorch reference for compare...")
        ref_model = OmniVoice.from_pretrained(
            str(model_dir),
            dtype=torch.float32,
            device_map="cuda",
            train=True,
        ).eval()
        _round_model_float_weights_to_bfloat16(ref_model)
        compare_refs = reference_setup.build_compare_references(
            ref_model,
            batch_input_ids=batch_input_ids,
            batch_audio_mask=batch_audio_mask,
            attention_mask=attn_mask_np,
            tokens=tokens,
            audio_mask_id=audio_mask_id,
            rng_seed=rng_seed,
        )

    rt.initialize_request_state(
        {
            model_tensors().batch_input_ids: batch_input_ids,
            model_tensors().batch_audio_mask: batch_audio_mask,
            model_tensors().attention_mask: attn_mask_np,
            model_tensors().audio_mask_id: np.array([audio_mask_id], dtype=np.int64),
            model_tensors().rng_seed: np.array([rng_seed], dtype=np.uint32),
            model_tensors().tokens: tokens,
        }
    )

    # Compute RoPE once on GPU (positions are fixed for masked decoding)
    rt.register_inputs({
        model_tensors().rope.start_position: np.array([0], dtype=np.int64),
        model_tensors().rope.theta: np.array([1_000_000.0], dtype=np.float32),
    })
    dispatch.run_rope_table(rt, frame_name="omnivoice.rope")

    unmasked = 0
    generation_replay_plan: ReplayPlan | None = None
    use_replay = compare_refs is None and num_steps > 1
    for step in range(num_steps):
        k = schedule[step]
        if k <= 0:
            continue

        step_inputs = _generation_step_inputs(step, k)
        if generation_replay_plan is None:
            rt.register_inputs(step_inputs)
            if compare_refs is not None:
                _run_generation_step_with_compare(
                    rt,
                    step=step,
                    unmask_count=k,
                    refs=compare_refs,
                )
            else:
                dispatch.run_generation_step(rt, step=step)
            if use_replay:
                generation_replay_plan = _build_generation_replay_plan(
                    rt,
                    frame=f"omnivoice.step.{step:04d}",
                )
        else:
            stage_replay_step_inputs(
                rt,
                plan=generation_replay_plan,
                inputs=step_inputs,
                write_through=(
                    model_tensors().step_index,
                    model_tensors().unmask_count,
                ),
            )
            execute_replay(generation_replay_plan)
        unmasked += k

        if step % 8 == 0 or step == num_steps - 1:
            total = num_audio_codebook * target_len
            print(f"  Step {step}: unmasked {unmasked}/{total} ({100*unmasked/total:.0f}%)")

    # Decode audio tokens
    print("\nDecoding audio tokens...")
    generated_tokens = rt.read_request_state(model_tensors().tokens)
    rt.close()
    audio_tokenizer_path = model_dir / "audio_tokenizer"
    audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
        str(audio_tokenizer_path), device_map="cuda"
    )
    audio_output = cast(
        _AudioDecodeOutput,
        audio_tokenizer.decode(torch.from_numpy(generated_tokens).cuda()),
    )
    waveform = audio_output.audio_values[0].cpu()

    # Save wav
    output_path = save_audio_wav(waveform, output_path)
    print(f"\nOutput: {output_path}")
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
