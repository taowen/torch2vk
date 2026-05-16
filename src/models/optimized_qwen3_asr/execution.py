"""Qwen3-ASR input preparation shared by generated and optimized runtimes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, TypeGuard

import numpy as np
import numpy.typing as npt
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
from qwen_asr.inference.utils import (
    normalize_audios,
    normalize_language_name,
    validate_language,
)

from models.hf_cache import resolve_cached_model
from models.optimized_qwen3_asr.pytorch.example import REPO_ID

QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS = (151645, 151643)


class Qwen3AsrProcessorLike(Protocol):
    def __call__(
        self,
        *,
        text: Sequence[str],
        audio: Sequence[np.ndarray],
        return_tensors: str,
        padding: bool,
    ) -> Mapping[str, object]: ...

    def apply_chat_template(
        self,
        messages: object,
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str: ...

    def batch_decode(
        self,
        sequences: object,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]: ...


class _TorchArrayLike(Protocol):
    def detach(self) -> "_TorchArrayLike": ...

    def cpu(self) -> "_TorchArrayLike": ...

    def numpy(self) -> np.ndarray: ...


class _KeywordCallable(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class Qwen3AsrPreparedInputs:
    prompt: str
    input_ids: np.ndarray
    attention_mask: np.ndarray
    input_features: np.ndarray
    feature_attention_mask: np.ndarray

    @property
    def prompt_length(self) -> int:
        return int(self.input_ids.shape[-1])

    @property
    def audio_feature_length(self) -> int:
        return int(self.feature_attention_mask.sum(axis=-1)[0])


def prepare_qwen3_asr_inputs(
    *,
    model_dir: str | Path | None,
    wav: str | Path | np.ndarray,
    language: str | None = "English",
    context: str = "",
) -> tuple[Qwen3AsrProcessorLike, Qwen3AsrPreparedInputs]:
    resolved_model_dir = resolve_cached_model(REPO_ID, model_dir)
    processor = _cached_processor(str(resolved_model_dir))
    force_language = _normalize_optional_language(
        language,
        normalize_language_name=normalize_language_name,
        validate_language=validate_language,
    )
    waveform = (
        np.asarray(wav, dtype=np.float32)
        if isinstance(wav, np.ndarray)
        else normalize_audios(str(wav))[0]
    )
    prompt = build_qwen3_asr_text_prompt(
        processor=processor,
        context=context,
        force_language=force_language,
    )
    batch = processor(text=[prompt], audio=[waveform], return_tensors="pt", padding=True)
    return processor, Qwen3AsrPreparedInputs(
        prompt=prompt,
        input_ids=_to_numpy(batch["input_ids"], dtype=np.int64),
        attention_mask=_to_numpy(batch["attention_mask"], dtype=np.int64),
        input_features=_to_numpy(batch["input_features"], dtype=np.float32),
        feature_attention_mask=_to_numpy(batch["feature_attention_mask"], dtype=np.int64),
    )


@lru_cache(maxsize=2)
def _cached_processor(resolved_model_dir: str) -> Qwen3AsrProcessorLike:
    return _Qwen3AsrProcessorAdapter(
        Qwen3ASRProcessor.from_pretrained(resolved_model_dir, fix_mistral_regex=True)
    )


def build_qwen3_asr_text_prompt(
    *,
    processor: Qwen3AsrProcessorLike,
    context: str = "",
    force_language: str | None = "English",
) -> str:
    messages = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if force_language:
        prompt = f"{prompt}language {force_language}<asr_text>"
    return str(prompt)


def _normalize_optional_language(
    language: str | None,
    *,
    normalize_language_name: Callable[[str], str],
    validate_language: Callable[[str], object],
) -> str | None:
    if language is None or str(language).strip() == "":
        return None
    normalized = normalize_language_name(str(language))
    validate_language(normalized)
    return str(normalized)


def _to_numpy(value: object, *, dtype: npt.DTypeLike) -> np.ndarray:
    array = value.detach().cpu().numpy() if _is_torch_array_like(value) else np.asarray(value)
    return np.asarray(array, dtype=dtype)


@dataclass(frozen=True, slots=True)
class _Qwen3AsrProcessorAdapter:
    raw: object

    def __call__(
        self,
        *,
        text: Sequence[str],
        audio: Sequence[np.ndarray],
        return_tensors: str,
        padding: bool,
    ) -> Mapping[str, object]:
        if not _is_keyword_callable(self.raw):
            raise TypeError(f"processor must be callable, got {type(self.raw).__name__}")
        batch = self.raw(
            text=list(text),
            audio=list(audio),
            return_tensors=return_tensors,
            padding=padding,
        )
        if not isinstance(batch, Mapping):
            raise TypeError(f"processor returned {type(batch).__name__}, expected mapping")
        return batch

    def apply_chat_template(
        self,
        messages: object,
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        method = getattr(self.raw, "apply_chat_template", None)
        if not _is_keyword_callable(method):
            raise TypeError("processor.apply_chat_template must be callable")
        prompt = method(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
        )
        return str(prompt)

    def batch_decode(
        self,
        sequences: object,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        method = getattr(self.raw, "batch_decode", None)
        if not _is_keyword_callable(method):
            raise TypeError("processor.batch_decode must be callable")
        decoded = method(
            sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        if not isinstance(decoded, Sequence) or isinstance(decoded, str | bytes):
            raise TypeError(f"processor.batch_decode returned {type(decoded).__name__}")
        return [str(item) for item in decoded]


def _is_torch_array_like(value: object) -> TypeGuard[_TorchArrayLike]:
    detach = getattr(value, "detach", None)
    cpu = getattr(value, "cpu", None)
    numpy = getattr(value, "numpy", None)
    return callable(detach) and callable(cpu) and callable(numpy)


def _is_keyword_callable(value: object) -> TypeGuard[_KeywordCallable]:
    return callable(value)
