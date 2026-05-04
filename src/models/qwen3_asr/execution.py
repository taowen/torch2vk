"""Qwen3-ASR runtime orchestration entry points."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeGuard

import numpy as np
import numpy.typing as npt

from models.hf_cache import resolve_cached_model
from models.qwen3_asr.audio_tower import (
    run_qwen3_asr_audio_tower as run_qwen3_asr_audio_tower,
)
from models.qwen3_asr.pytorch.example import REPO_ID
from models.qwen3_asr.rope import precompute_qwen3_asr_mrope
from models.qwen3_asr.text_decode import run_qwen3_asr_text_decode
from models.qwen3_asr.text_prefill import run_qwen3_asr_text_prefill
from models.qwen3_asr.tensors.text import (
    Qwen3AsrTextDecodeTensors,
    Qwen3AsrTextPrefillTensors,
    Qwen3AsrTextTensors,
)
from models.qwen3_asr.token_select import run_qwen3_asr_token_select
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession

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
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
    from qwen_asr.inference.utils import (
        normalize_audios,
        normalize_language_name,
        validate_language,
    )

    resolved_model_dir = resolve_cached_model(REPO_ID, model_dir)
    processor = _Qwen3AsrProcessorAdapter(
        Qwen3ASRProcessor.from_pretrained(str(resolved_model_dir), fix_mistral_regex=True)
    )
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


def run_qwen3_asr_greedy_decode_loop(
    rt: RuntimeSession,
    tensors: Qwen3AsrTextTensors,
    *,
    max_new_tokens: int,
    rope_theta: float = 5_000_000.0,
    mrope_section: tuple[int, ...] = (24, 20, 20),
) -> LogicalTensor:
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    prefill = tensors.prefill
    head_dim = prefill.rope_cos.concrete_shape[-1]

    _register_prefill_rope(rt, prefill, rope_theta=rope_theta, mrope_section=mrope_section)
    run_qwen3_asr_text_prefill(rt, prefill)
    run_qwen3_asr_token_select(rt, tensors.token_select, logits=prefill.logits)
    _grow_generated_tokens(rt, tensors, generated_length=1)

    prompt_length = prefill.input_ids.concrete_shape[-1]
    decode = tensors.decode
    for step in range(max_new_tokens - 1):
        next_token = rt.read_request_state(tensors.token_select.next_token)
        _grow_kv_caches_for_next_decode_step(rt, tensors)
        _register_decode_rope(
            rt,
            decode,
            cache_position=prompt_length + step,
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )
        rt.register_inputs({decode.input_ids: next_token.reshape(1, 1)})
        run_qwen3_asr_text_decode(rt, decode, step=step)
        run_qwen3_asr_token_select(rt, tensors.token_select, logits=decode.logits)
        _grow_generated_tokens(rt, tensors, generated_length=step + 2)
    return tensors.token_select.generated_tokens


def _register_prefill_rope(
    rt: RuntimeSession,
    prefill: Qwen3AsrTextPrefillTensors,
    *,
    rope_theta: float,
    mrope_section: tuple[int, ...],
) -> None:
    prompt_length = prefill.rope_cos.concrete_shape[1]
    head_dim = prefill.rope_cos.concrete_shape[2]
    position_ids = np.arange(prompt_length, dtype=np.int64)[np.newaxis, :]
    position_ids_3d = np.broadcast_to(position_ids[np.newaxis, :, :], (3, 1, prompt_length)).copy()
    cos, sin = precompute_qwen3_asr_mrope(
        position_ids=position_ids_3d,
        head_dim=head_dim,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
    )
    rt.register_inputs({prefill.rope_cos: cos, prefill.rope_sin: sin})


def _register_decode_rope(
    rt: RuntimeSession,
    decode: Qwen3AsrTextDecodeTensors,
    *,
    cache_position: int,
    head_dim: int,
    rope_theta: float,
    mrope_section: tuple[int, ...],
) -> None:
    position_ids_3d = np.full((3, 1, 1), cache_position, dtype=np.int64)
    cos, sin = precompute_qwen3_asr_mrope(
        position_ids=position_ids_3d,
        head_dim=head_dim,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
    )
    rt.register_inputs({decode.rope_cos: cos, decode.rope_sin: sin})


def _grow_kv_caches_for_next_decode_step(rt: RuntimeSession, tensors: Qwen3AsrTextTensors) -> None:
    for layer in tensors.decode.layers:
        for cache in (layer.key_cache, layer.value_cache):
            batch, heads, cache_length, head_dim = cache.concrete_shape
            rt.grow_request_state(cache, (batch, heads, cache_length + 1, head_dim))


def _grow_generated_tokens(
    rt: RuntimeSession,
    tensors: Qwen3AsrTextTensors,
    *,
    generated_length: int,
) -> None:
    batch, _length = tensors.token_select.generated_tokens.concrete_shape
    rt.grow_request_state(tensors.token_select.generated_tokens, (batch, generated_length))


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
