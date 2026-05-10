"""Qwen3-ASR greedy token selection shader."""

from __future__ import annotations

from torch2vk.export.shaders.qwen3_asr_token_select_f32 import (
    QWEN3_ASR_TOKEN_SELECT_GREEDY_F32,
)

__all__ = ["QWEN3_ASR_TOKEN_SELECT_GREEDY_F32"]
