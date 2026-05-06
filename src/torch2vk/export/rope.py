"""RoPE helpers shared by model exporters/adapters."""

from __future__ import annotations

import numpy as np


def precompute_qwen3_asr_mrope(
    *,
    position_ids: np.ndarray,
    head_dim: int,
    rope_theta: float = 5_000_000.0,
    mrope_section: tuple[int, ...] = (24, 20, 20),
    attention_scaling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Qwen3-ASR multimodal RoPE cos/sin tables."""
    half_dim = head_dim // 2
    if sum(mrope_section) != half_dim:
        raise ValueError(
            f"mrope_section {mrope_section} must sum to head_dim/2={half_dim}"
        )
    if position_ids.shape[0] != 3 or position_ids.shape[1] != 1:
        raise ValueError(
            f"position_ids must have shape (3, 1, seq_len), got {position_ids.shape}"
        )

    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    inv_freq_3d = inv_freq[np.newaxis, np.newaxis, :, np.newaxis]
    pos_3d = position_ids[:, :, np.newaxis, :].astype(np.float64)
    freqs = (inv_freq_3d * pos_3d).transpose(0, 1, 3, 2)
    freqs_t = _apply_interleaved_mrope(freqs, mrope_section)
    emb = np.concatenate([freqs_t, freqs_t], axis=-1)
    cos = (np.cos(emb) * attention_scaling).astype(np.float32)
    sin = (np.sin(emb) * attention_scaling).astype(np.float32)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


def _apply_interleaved_mrope(freqs: np.ndarray, mrope_section: tuple[int, ...]) -> np.ndarray:
    freqs_t = freqs[0].copy()
    for dim_idx, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim_idx] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim_idx, ..., idx]
    return freqs_t
