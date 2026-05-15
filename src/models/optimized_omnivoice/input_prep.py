"""Host-side OmniVoice input preparation for optimized OmniVoice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from omnivoice.models.omnivoice import OmniVoiceConfig, _combine_text, _tokenize_with_nonverbal_tags
from omnivoice.utils.duration import RuleDurationEstimator
from omnivoice.utils.text import chunk_text_punctuation


DEFAULT_TEXT = "hello world this is a speech recognition test"
SEQ_CAPACITY = 256
TARGET_CAPACITY = 128
OMNIVOICE_FRAME_RATE = 25
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
    cond_target_start: int


@dataclass(frozen=True, slots=True)
class OmniVoiceTextChunk:
    text: str
    target_len: int


class _TokenizedText(Protocol):
    input_ids: torch.Tensor


class OmniVoiceTokenizer(Protocol):
    def __call__(self, text: str, *, return_tensors: str) -> _TokenizedText: ...


def estimate_target_len(
    text: str,
    *,
    ref_text: str | None = None,
    ref_audio_tokens: np.ndarray | None = None,
) -> int:
    estimator = RuleDurationEstimator()
    estimate = estimator.estimate_duration(
        text,
        _FALLBACK_REF_TEXT if ref_text is None else ref_text,
        _FALLBACK_REF_TOKENS if ref_audio_tokens is None else int(ref_audio_tokens.shape[-1]),
    )
    return max(1, int(estimate))


def chunk_omnivoice_text_for_capacity(
    text: str,
    *,
    tokenizer: OmniVoiceTokenizer,
    config: OmniVoiceConfig,
    seq_capacity: int = SEQ_CAPACITY,
    target_capacity: int = TARGET_CAPACITY,
) -> tuple[OmniVoiceTextChunk, ...]:
    text = text.strip()
    if not text:
        raise ValueError("OmniVoice text is empty")

    last_error: str | None = None
    budget_step = max(8, target_capacity // 8)
    min_budget = max(16, target_capacity // 2)
    for target_budget in range(target_capacity, min_budget - 1, -budget_step):
        chunks = _candidate_text_chunks(text, target_budget=target_budget)
        valid_chunks, last_error = _validate_text_chunks(
            chunks,
            tokenizer=tokenizer,
            config=config,
            seq_capacity=seq_capacity,
            target_capacity=target_capacity,
        )
        if valid_chunks is not None:
            return valid_chunks

    raise ValueError(
        "OmniVoice text cannot be chunked into the fixed topology "
        f"(seq_capacity={seq_capacity}, target_capacity={target_capacity}): {last_error}"
    )


def _candidate_text_chunks(text: str, *, target_budget: int) -> tuple[str, ...]:
    target_len = estimate_target_len(text)
    if target_len <= target_budget:
        return (text,)
    avg_tokens_per_char = target_len / len(text)
    chunk_len = max(3, int(target_budget / avg_tokens_per_char))
    punctuation_chunks = chunk_text_punctuation(text=text, chunk_len=chunk_len, min_chunk_len=3)
    if not punctuation_chunks:
        raise ValueError("OmniVoice text chunking produced no chunks")

    chunks: list[str] = []
    for chunk in punctuation_chunks:
        if estimate_target_len(chunk) <= target_budget:
            chunks.append(chunk)
        else:
            chunks.extend(_split_text_chunk_by_words(chunk, target_budget=target_budget))
    return tuple(chunks)


def _split_text_chunk_by_words(text: str, *, target_budget: int) -> tuple[str, ...]:
    words = text.split()
    if not words:
        return ()

    chunks: list[list[str]] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join((*current, word))
        if current and estimate_target_len(candidate) > target_budget:
            chunks.append(current)
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(current)

    if len(chunks) > 1:
        _rebalance_short_tail(chunks, target_budget=target_budget)
    return tuple(" ".join(chunk) for chunk in chunks)


def _rebalance_short_tail(chunks: list[list[str]], *, target_budget: int) -> None:
    while len(chunks[-1]) > 0 and estimate_target_len(" ".join(chunks[-1])) < target_budget // 2:
        previous = chunks[-2]
        if len(previous) <= 1:
            return
        moved = previous.pop()
        chunks[-1].insert(0, moved)
        if estimate_target_len(" ".join(chunks[-1])) > target_budget:
            chunks[-1].pop(0)
            previous.append(moved)
            return


def _validate_text_chunks(
    chunks: tuple[str, ...],
    *,
    tokenizer: OmniVoiceTokenizer,
    config: OmniVoiceConfig,
    seq_capacity: int,
    target_capacity: int,
) -> tuple[tuple[OmniVoiceTextChunk, ...] | None, str | None]:
    first_target_len: int | None = None
    valid_chunks: list[OmniVoiceTextChunk] = []
    for idx, chunk in enumerate(chunks):
        ref_text = None if idx == 0 else chunks[0]
        ref_audio_token_len = None
        if idx > 0:
            if first_target_len is None:
                raise RuntimeError("first_target_len must be set after validating chunk 0")
            ref_audio_token_len = first_target_len
        target_len, seq_len = _measure_chunk_shape(
            text=chunk,
            tokenizer=tokenizer,
            config=config,
            ref_text=ref_text,
            ref_audio_token_len=ref_audio_token_len,
        )
        if target_len > target_capacity:
            return (
                None,
                f"OmniVoice chunk target_len={target_len} exceeds "
                f"TARGET_CAPACITY={target_capacity}",
            )
        if seq_len > seq_capacity:
            return (
                None,
                f"OmniVoice chunk seq_len={seq_len} exceeds SEQ_CAPACITY={seq_capacity}",
            )
        if idx == 0:
            first_target_len = target_len
        valid_chunks.append(OmniVoiceTextChunk(text=chunk, target_len=target_len))
    return tuple(valid_chunks), None


def _measure_chunk_shape(
    *,
    text: str,
    tokenizer: OmniVoiceTokenizer,
    config: OmniVoiceConfig,
    ref_text: str | None,
    ref_audio_token_len: int | None,
) -> tuple[int, int]:
    style_text = ""
    if ref_audio_token_len is not None:
        style_text += "<|denoise|>"
    style_text += "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_len = int(tokenizer(style_text, return_tensors="pt").input_ids.shape[1])

    wrapped_text = f"<|text_start|>{_combine_text(text, ref_text=ref_text)}<|text_end|>"
    text_len = int(_tokenize_with_nonverbal_tags(wrapped_text, tokenizer).shape[1])
    target_len = estimate_target_len(
        text,
        ref_text=ref_text,
        ref_audio_tokens=(
            None
            if ref_audio_token_len is None
            else np.zeros((config.num_audio_codebook, ref_audio_token_len), dtype=np.int64)
        ),
    )
    seq_len = style_len + text_len + (0 if ref_audio_token_len is None else ref_audio_token_len) + target_len
    return target_len, seq_len


def prepare_omnivoice_inputs(
    *,
    text: str,
    tokenizer: OmniVoiceTokenizer,
    config: OmniVoiceConfig,
    ref_text: str | None = None,
    ref_audio_tokens: np.ndarray | None = None,
    seq_capacity: int = SEQ_CAPACITY,
    target_capacity: int = TARGET_CAPACITY,
) -> PreparedOmniVoiceInputs:
    num_audio_codebook = config.num_audio_codebook
    audio_mask_id = config.audio_mask_id

    style_text = ""
    if ref_audio_tokens is not None:
        style_text += "<|denoise|>"
    style_text += "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_text, return_tensors="pt").input_ids.numpy().astype(np.int64)
    style_tokens = np.broadcast_to(
        style_ids,
        (num_audio_codebook, style_ids.shape[1]),
    )[None, :, :].copy()

    wrapped_text = f"<|text_start|>{_combine_text(text, ref_text=ref_text)}<|text_end|>"
    text_ids = _tokenize_with_nonverbal_tags(wrapped_text, tokenizer).numpy().astype(np.int64)
    text_tokens = np.broadcast_to(
        text_ids,
        (num_audio_codebook, text_ids.shape[1]),
    )[None, :, :].copy()

    ref_tokens: np.ndarray | None = None
    if ref_audio_tokens is not None:
        ref_tokens = np.asarray(ref_audio_tokens, dtype=np.int64)
        if ref_tokens.shape[0] != num_audio_codebook:
            raise ValueError(
                f"ref_audio_tokens has {ref_tokens.shape[0]} codebooks, "
                f"expected {num_audio_codebook}"
            )
        ref_tokens = ref_tokens[None, :, :]

    target_len = estimate_target_len(
        text,
        ref_text=ref_text,
        ref_audio_tokens=None if ref_tokens is None else ref_tokens[0],
    )
    if target_len > target_capacity:
        raise ValueError(
            f"OmniVoice chunk target_len={target_len} exceeds TARGET_CAPACITY={target_capacity}"
        )
    target_audio_tokens = np.full(
        (1, num_audio_codebook, target_len),
        audio_mask_id,
        dtype=np.int64,
    )
    parts = [style_tokens, text_tokens]
    if ref_tokens is not None:
        parts.append(ref_tokens)
    parts.append(target_audio_tokens)
    cond_input_ids = np.concatenate(parts, axis=2)
    cond_total_len = cond_input_ids.shape[2]
    if cond_total_len > seq_capacity:
        raise ValueError(
            f"OmniVoice chunk seq_len={cond_total_len} exceeds SEQ_CAPACITY={seq_capacity}"
        )
    cond_target_start = cond_total_len - target_len
    cond_audio_start = cond_target_start
    if ref_tokens is not None:
        cond_audio_start -= ref_tokens.shape[-1]

    cond_audio_mask = np.zeros((1, cond_total_len), dtype=np.uint32)
    cond_audio_mask[0, cond_audio_start:] = 1

    batch_size = 1
    batch_input_ids = np.full(
        (2 * batch_size, num_audio_codebook, seq_capacity),
        audio_mask_id,
        dtype=np.int64,
    )
    batch_audio_mask = np.zeros((2 * batch_size, seq_capacity), dtype=np.uint32)
    batch_attention_mask = np.zeros(
        (2 * batch_size, 1, seq_capacity, seq_capacity),
        dtype=np.bool_,
    )

    batch_input_ids[0, :, :cond_total_len] = cond_input_ids[0]
    batch_audio_mask[0, :cond_total_len] = cond_audio_mask[0]
    batch_attention_mask[0, :, :cond_total_len, :cond_total_len] = True
    if seq_capacity > cond_total_len:
        pad_diag = np.arange(cond_total_len, seq_capacity)
        batch_attention_mask[0, :, pad_diag, pad_diag] = True

    batch_input_ids[batch_size, :, :target_len] = cond_input_ids[0, :, -target_len:]
    batch_audio_mask[batch_size, :target_len] = cond_audio_mask[0, -target_len:]
    batch_attention_mask[batch_size, :, :target_len, :target_len] = True
    if seq_capacity > target_len:
        pad_diag = np.arange(target_len, seq_capacity)
        batch_attention_mask[batch_size, :, pad_diag, pad_diag] = True

    attention_mask = np.zeros(batch_attention_mask.shape, dtype=np.float32)
    attention_mask[~batch_attention_mask] = -np.finfo(np.float32).max

    return PreparedOmniVoiceInputs(
        batch_input_ids=batch_input_ids,
        batch_audio_mask=batch_audio_mask,
        attention_mask=attention_mask,
        target_len=target_len,
        seq_len=cond_total_len,
        cond_audio_start=cond_audio_start,
        cond_target_start=cond_target_start,
    )
