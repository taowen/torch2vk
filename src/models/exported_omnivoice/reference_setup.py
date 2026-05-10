"""Generated PyTorch reference setup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.exported_omnivoice.pytorch_modules import (
    AudioHeadReference,
    InputEmbedReference,
    LlmForwardReference,
    TokenScoreReference,
    TokenUpdateReference,
)
from omnivoice.models.omnivoice import OmniVoice


@dataclass(slots=True)
class CompareReferences:
    input_embed: InputEmbedReference
    llm_forward: LlmForwardReference
    audio_head: AudioHeadReference
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


def build_compare_references(
    model: OmniVoice,
    *,
    batch_input_ids: np.ndarray,
    batch_audio_mask: np.ndarray,
    attention_mask: np.ndarray,
    tokens: np.ndarray,
    audio_mask_id: int,
    rng_seed: int,
) -> CompareReferences:
    rope_cos, rope_sin = _make_rope_table(
        batch=attention_mask.shape[0],
        sequence_length=attention_mask.shape[-1],
        head_dim=128,
        theta=1_000_000.0,
    )
    return CompareReferences(
        input_embed=InputEmbedReference(model),
        llm_forward=LlmForwardReference(model),
        audio_head=AudioHeadReference(model),
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
