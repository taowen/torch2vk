"""Host-side OmniVoice input preparation shared by export and run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from omnivoice.models.omnivoice import OmniVoiceConfig, _combine_text, _tokenize_with_nonverbal_tags
from omnivoice.utils.duration import RuleDurationEstimator


DEFAULT_TEXT = "hello world this is a speech recognition test"
_FALLBACK_REF_TEXT = "Nice to meet you."
_FALLBACK_REF_TOKENS = 25


@dataclass(frozen=True, slots=True)
class PreparedOmniVoiceInputs:
    batch_input_ids: np.ndarray
    batch_audio_mask: np.ndarray
    attention_mask: np.ndarray
    target_len: int
    seq_len: int
    cond_audio_start: int


class _TokenizedText(Protocol):
    input_ids: torch.Tensor


class OmniVoiceTokenizer(Protocol):
    def __call__(self, text: str, *, return_tensors: str) -> _TokenizedText: ...


def estimate_target_len(text: str) -> int:
    estimator = RuleDurationEstimator()
    estimate = estimator.estimate_duration(
        text,
        _FALLBACK_REF_TEXT,
        _FALLBACK_REF_TOKENS,
    )
    return max(1, int(estimate))


def prepare_omnivoice_inputs(
    *,
    text: str,
    tokenizer: OmniVoiceTokenizer,
    config: OmniVoiceConfig,
) -> PreparedOmniVoiceInputs:
    num_audio_codebook = config.num_audio_codebook
    audio_mask_id = config.audio_mask_id

    style_text = "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_text, return_tensors="pt").input_ids.numpy().astype(np.int64)
    style_tokens = np.broadcast_to(
        style_ids,
        (num_audio_codebook, style_ids.shape[1]),
    )[None, :, :].copy()

    wrapped_text = f"<|text_start|>{_combine_text(text)}<|text_end|>"
    text_ids = _tokenize_with_nonverbal_tags(wrapped_text, tokenizer).numpy().astype(np.int64)
    text_tokens = np.broadcast_to(
        text_ids,
        (num_audio_codebook, text_ids.shape[1]),
    )[None, :, :].copy()

    target_len = estimate_target_len(text)
    target_audio_tokens = np.full(
        (1, num_audio_codebook, target_len),
        audio_mask_id,
        dtype=np.int64,
    )
    cond_input_ids = np.concatenate([style_tokens, text_tokens, target_audio_tokens], axis=2)
    cond_total_len = cond_input_ids.shape[2]
    cond_audio_start = cond_total_len - target_len

    cond_audio_mask = np.zeros((1, cond_total_len), dtype=np.uint32)
    cond_audio_mask[0, cond_audio_start:] = 1

    batch_size = 1
    seq_len = cond_total_len
    batch_input_ids = np.full(
        (2 * batch_size, num_audio_codebook, seq_len),
        audio_mask_id,
        dtype=np.int64,
    )
    batch_audio_mask = np.zeros((2 * batch_size, seq_len), dtype=np.uint32)
    batch_attention_mask = np.zeros((2 * batch_size, 1, seq_len, seq_len), dtype=np.bool_)

    batch_input_ids[0, :, :cond_total_len] = cond_input_ids[0]
    batch_audio_mask[0, :cond_total_len] = cond_audio_mask[0]
    batch_attention_mask[0, :, :cond_total_len, :cond_total_len] = True

    batch_input_ids[batch_size, :, :target_len] = cond_input_ids[0, :, -target_len:]
    batch_audio_mask[batch_size, :target_len] = cond_audio_mask[0, -target_len:]
    batch_attention_mask[batch_size, :, :target_len, :target_len] = True
    if seq_len > target_len:
        pad_diag = np.arange(target_len, seq_len)
        batch_attention_mask[batch_size, :, pad_diag, pad_diag] = True

    attention_mask = np.zeros(batch_attention_mask.shape, dtype=np.float32)
    attention_mask[~batch_attention_mask] = -np.finfo(np.float32).max

    return PreparedOmniVoiceInputs(
        batch_input_ids=batch_input_ids,
        batch_audio_mask=batch_audio_mask,
        attention_mask=attention_mask,
        target_len=target_len,
        seq_len=seq_len,
        cond_audio_start=cond_audio_start,
    )
