"""PyTorch/Vulkan comparison entry points for exported OmniVoice."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from transformers import AutoTokenizer

from models.exported_omnivoice import reference
from models.exported_omnivoice.dispatch.audio_head import run_audio_head
from models.exported_omnivoice.dispatch.llm_forward import run_llm_forward
from models.exported_omnivoice.input_prep import DEFAULT_TEXT, prepare_omnivoice_inputs
from models.exported_omnivoice.pytorch_modules import (
    InputEmbedReference,
    LlmForwardReference,
    TokenScoreReference,
    TokenUpdateReference,
)
from models.exported_omnivoice.run import (
    _generation_step_inputs,
    _get_time_steps,
    _run_input_embed,
    _run_rope_table,
    _run_token_score,
    _run_token_update,
)
from models.exported_omnivoice.shaders.registry import get_shader
from models.exported_omnivoice.tensors.model import create_model_tensors, model_tensors
from models.hf_cache import resolve_cached_model
from models.optimized_omnivoice.pytorch.example import REPO_ID
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession


@dataclass(slots=True)
class _OmniVoiceCompareState:
    input_embed: InputEmbedReference
    llm_forward: LlmForwardReference
    token_score: TokenScoreReference
    token_update: TokenUpdateReference
    batch_input_ids: torch.Tensor
    batch_audio_mask: torch.Tensor
    attention_mask: torch.Tensor
    tokens: torch.Tensor
    audio_mask_id: torch.Tensor
    rng_seed: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor


def _make_rope_table(
    *,
    batch: int,
    sequence_length: int,
    head_dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = torch.arange(sequence_length, device="cuda", dtype=torch.float32).view(1, -1, 1)
    dim = torch.arange(head_dim, device="cuda", dtype=torch.float32).view(1, 1, -1)
    half_dim = head_dim // 2
    freq_idx = torch.remainder(dim, half_dim)
    inv_freq = torch.pow(
        torch.tensor(theta, device="cuda", dtype=torch.float32),
        -(2.0 * freq_idx) / head_dim,
    )
    angle = token * inv_freq
    cos = torch.cos(angle).expand(batch, -1, -1).contiguous()
    sin = torch.sin(angle).expand(batch, -1, -1).contiguous()
    return cos, sin


def _build_compare_references(
    model: OmniVoice,
    *,
    batch_input_ids: np.ndarray,
    batch_audio_mask: np.ndarray,
    attention_mask: np.ndarray,
    tokens: np.ndarray,
    audio_mask_id: int,
    rng_seed: int,
    head_dim: int,
) -> _OmniVoiceCompareState:
    reference.set_model(model)
    rope_cos, rope_sin = _make_rope_table(
        batch=attention_mask.shape[0],
        sequence_length=attention_mask.shape[-1],
        head_dim=head_dim,
        theta=1_000_000.0,
    )
    return _OmniVoiceCompareState(
        input_embed=InputEmbedReference(model),
        llm_forward=LlmForwardReference(model),
        token_score=TokenScoreReference(model),
        token_update=TokenUpdateReference(),
        batch_input_ids=torch.from_numpy(np.ascontiguousarray(batch_input_ids)).cuda(),
        batch_audio_mask=torch.from_numpy(
            np.ascontiguousarray(batch_audio_mask.astype(np.bool_))
        ).cuda(),
        attention_mask=torch.from_numpy(np.ascontiguousarray(attention_mask)).cuda(),
        tokens=torch.from_numpy(np.ascontiguousarray(tokens)).cuda(),
        audio_mask_id=torch.tensor([audio_mask_id], dtype=torch.int64, device="cuda"),
        rng_seed=torch.tensor([rng_seed], dtype=torch.int64, device="cuda"),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )


def _vulkan_tensor(rt: RuntimeSession, tensor: LogicalTensor) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rt.readback(tensor))).cuda()


def _run_generation_step_with_compare(
    rt: RuntimeSession,
    *,
    step: int,
    unmask_count: int,
    refs: _OmniVoiceCompareState,
) -> None:
    with rt.frame(f"omnivoice.step.{step:04d}"):
        _run_input_embed(rt)
        reference.run_input_embed(
            rt,
            refs.input_embed,
            step=step,
            input_ids=refs.batch_input_ids,
            audio_mask=refs.batch_audio_mask,
        )
        hidden_states = _vulkan_tensor(rt, model_tensors().llm_forward.hidden_states).float()

        run_llm_forward(rt)
        reference.run_llm_forward(
            rt,
            refs.llm_forward,
            step=step,
            hidden_states=hidden_states,
            cos=refs.rope_cos,
            sin=refs.rope_sin,
            attention_mask=refs.attention_mask,
        )
        llm_output = _vulkan_tensor(rt, model_tensors().llm_forward.mul_365).float()

        run_audio_head(rt)
        reference.run_audio_head(
            rt,
            step=step,
            input=llm_output,
        )
        logits = _vulkan_tensor(rt, model_tensors().audio_head.linear).float()

        step_index = torch.tensor([step], dtype=torch.int64, device="cuda")
        _run_token_score(rt)
        reference.run_token_score(
            rt,
            refs.token_score,
            step=step,
            logits=logits,
            tokens=refs.tokens,
            audio_mask_id=refs.audio_mask_id,
            rng_seed=refs.rng_seed,
            step_index=step_index,
        )
        candidate_tokens = _vulkan_tensor(rt, model_tensors().candidate_tokens).long()
        candidate_scores = _vulkan_tensor(rt, model_tensors().candidate_scores).float()

        _run_token_update(rt)
        unmask_count_t = torch.tensor([unmask_count], dtype=torch.uint32, device="cuda")
        reference.run_token_update(
            rt,
            refs.token_update,
            step=step,
            tokens=refs.tokens,
            batch_input_ids=refs.batch_input_ids,
            candidate_tokens=candidate_tokens,
            candidate_scores=candidate_scores,
            unmask_count=unmask_count_t,
        )
        refs.tokens = _vulkan_tensor(rt, model_tensors().tokens).long()
        refs.batch_input_ids = _vulkan_tensor(rt, model_tensors().batch_input_ids).long()


def compare_generation_steps(
    *,
    text: str = DEFAULT_TEXT,
    num_steps: int = 2,
) -> None:
    model_dir = resolve_cached_model(REPO_ID)
    config = OmniVoiceConfig(**json.loads((model_dir / "config.json").read_text()))
    llm_config = config.llm_config
    if llm_config is None:
        raise ValueError("OmniVoice config requires llm_config")

    text_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prepared = prepare_omnivoice_inputs(
        text=text,
        tokenizer=text_tokenizer,
        config=config,
    )
    create_model_tensors(target_len=prepared.target_len)
    expected_seq_len = model_tensors().batch_input_ids.spec.shape[2]
    if expected_seq_len != prepared.seq_len:
        raise ValueError(
            f"exported OmniVoice seq_len is {expected_seq_len}, "
            f"but prepared inputs require {prepared.seq_len}; regenerate exported_omnivoice"
        )

    model = cast(
        OmniVoice,
        OmniVoice.from_pretrained(
            str(model_dir),
            dtype=torch.float32,
            device_map="cuda",
            train=True,
        ).eval(),
    )
    tokens = np.full(
        (1, config.num_audio_codebook, prepared.target_len),
        config.audio_mask_id,
        dtype=np.int64,
    )
    rng_seed = 0x1234ABCD
    refs = _build_compare_references(
        model,
        batch_input_ids=prepared.batch_input_ids,
        batch_audio_mask=prepared.batch_audio_mask,
        attention_mask=prepared.attention_mask,
        tokens=tokens,
        audio_mask_id=config.audio_mask_id,
        rng_seed=rng_seed,
        head_dim=llm_config.head_dim,
    )

    rt = RuntimeSession.open(
        device_index=0,
        model_dir=model_dir,
        model_tensors=model_tensors(),
        get_shader=get_shader,
    )
    try:
        rt.initialize_request_state(
            {
                model_tensors().batch_input_ids: prepared.batch_input_ids,
                model_tensors().batch_audio_mask: prepared.batch_audio_mask,
                model_tensors().attention_mask: prepared.attention_mask,
                model_tensors().audio_mask_id: np.array([config.audio_mask_id], dtype=np.int64),
                model_tensors().rng_seed: np.array([rng_seed], dtype=np.uint32),
                model_tensors().tokens: tokens,
            }
        )
        rt.register_inputs({
            model_tensors().rope.start_position: np.array([0], dtype=np.int64),
            model_tensors().rope.theta: np.array([1_000_000.0], dtype=np.float32),
        })
        _run_rope_table(rt, frame_name="omnivoice.rope")

        timesteps = _get_time_steps(0.0, 1.0, num_steps, t_shift=0.1)
        total_mask = prepared.target_len * config.num_audio_codebook
        remaining = total_mask
        for step in range(num_steps):
            if step == num_steps - 1:
                unmask_count = remaining
            else:
                unmask_count = min(
                    math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                    remaining,
                )
            remaining -= int(unmask_count)
            if unmask_count <= 0:
                continue
            rt.register_inputs(_generation_step_inputs(step, int(unmask_count)))
            _run_generation_step_with_compare(
                rt,
                step=step,
                unmask_count=int(unmask_count),
                refs=refs,
            )
    finally:
        rt.close()
