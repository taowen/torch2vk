"""Compatibility wrapper for Qwen3-ASR RoPE helpers."""

from __future__ import annotations

import numpy as np

from torch2vk.export.rope import precompute_qwen3_asr_mrope as _precompute_qwen3_asr_mrope


def precompute_qwen3_asr_mrope(
    *,
    position_ids: np.ndarray,
    head_dim: int,
    rope_theta: float = 5_000_000.0,
    mrope_section: tuple[int, ...] = (24, 20, 20),
    attention_scaling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    return _precompute_qwen3_asr_mrope(
        position_ids=position_ids,
        head_dim=head_dim,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
        attention_scaling=attention_scaling,
    )
