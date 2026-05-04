"""Qwen3-ASR runtime orchestration entry points."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeGuard

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
    Qwen3AsrTokenSelectTensors,
)
from models.qwen3_asr.tensors.text_layer import Qwen3AsrTextLayerTensors
from models.qwen3_asr.token_select import run_qwen3_asr_token_select
from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    TensorSemantic,
    TensorSpec,
)
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.shader import ShaderVariant

if TYPE_CHECKING:
    from torch2vk.runtime.replay import ReplayPlan

QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS = (151645, 151643)
Qwen3AsrReplayMode = Literal["default", "require_cache", "force_record"]


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
    pytorch_compare: bool = True,
    stop_on_eos: bool = False,
) -> LogicalTensor:
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    prefill = tensors.prefill
    head_dim = prefill.rope_cos.concrete_shape[-1]

    _register_prefill_rope(rt, prefill, rope_theta=rope_theta, mrope_section=mrope_section)
    run_qwen3_asr_text_prefill(rt, prefill, pytorch_compare=pytorch_compare)
    run_qwen3_asr_token_select(rt, tensors.token_select, logits=prefill.logits)
    _append_generated_token(rt, tensors)
    if stop_on_eos and _token_select_done(rt, tensors):
        return tensors.token_select.generated_tokens

    prompt_length = prefill.input_ids.concrete_shape[-1]
    decode = tensors.decode
    for step in range(max_new_tokens - 1):
        next_token = rt.read_request_state(tensors.token_select.next_token)
        _ensure_kv_cache_capacity(rt, tensors, required_length=prompt_length + step + 1)
        _register_decode_rope(
            rt,
            decode,
            cache_position=prompt_length + step,
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )
        rt.register_inputs({decode.input_ids: next_token.reshape(1, 1)})
        run_qwen3_asr_text_decode(rt, decode, step=step, pytorch_compare=pytorch_compare)
        run_qwen3_asr_token_select(rt, tensors.token_select, logits=decode.logits)
        _append_generated_token(rt, tensors)
        if stop_on_eos and _token_select_done(rt, tensors):
            break
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
    rt.register_inputs(
        {
            decode.cache_position: np.array([cache_position], dtype=np.int64),
            decode.rope_cos: cos,
            decode.rope_sin: sin,
        }
    )


def _ensure_kv_cache_capacity(
    rt: RuntimeSession,
    tensors: Qwen3AsrTextTensors,
    *,
    required_length: int,
) -> None:
    for layer in tensors.decode.layers:
        for cache in (layer.key_cache, layer.value_cache):
            batch, heads, cache_length, head_dim = cache.concrete_shape
            if cache_length < required_length:
                rt.grow_request_state(cache, (batch, heads, required_length, head_dim))


def _append_generated_token(
    rt: RuntimeSession,
    tensors: Qwen3AsrTextTensors,
) -> None:
    generated = tensors.token_select.generated_tokens
    batch, length = generated.concrete_shape
    next_token = rt.read_request_state(tensors.token_select.next_token).reshape(batch, 1)
    previous = (
        np.empty((batch, 0), dtype=np.int64)
        if length == 0
        else rt.read_request_state(generated)
    )
    updated = np.ascontiguousarray(np.concatenate([previous, next_token], axis=1), dtype=np.int64)
    rt.grow_request_state(generated, updated.shape)
    rt.initialize_request_state({generated: updated})


def _token_select_done(rt: RuntimeSession, tensors: Qwen3AsrTextTensors) -> bool:
    done = rt.read_request_state(tensors.token_select.done)
    return bool(np.asarray(done).reshape(-1)[0])


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


def run_qwen3_asr_replay_decode_loop(
    rt: RuntimeSession,
    tensors: Qwen3AsrTextTensors,
    *,
    max_new_tokens: int,
    rope_theta: float = 5_000_000.0,
    mrope_section: tuple[int, ...] = (24, 20, 20),
    stop_on_eos: bool = True,
    mode: Qwen3AsrReplayMode = "default",
) -> LogicalTensor:
    """Decode loop using replay, with cached plans reused across compatible requests."""
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if mode not in {"default", "require_cache", "force_record"}:
        raise ValueError(
            "mode must be one of 'default', 'require_cache', or 'force_record', "
            f"got {mode!r}"
        )

    prefill = tensors.prefill
    decode = tensors.decode
    head_dim = prefill.rope_cos.concrete_shape[-1]
    prompt_length = prefill.input_ids.concrete_shape[-1]

    # Phase 1: Prefill (eager)
    _register_prefill_rope(rt, prefill, rope_theta=rope_theta, mrope_section=mrope_section)
    run_qwen3_asr_text_prefill(rt, prefill, pytorch_compare=False)
    run_qwen3_asr_token_select(rt, tensors.token_select, logits=prefill.logits)

    replay_generated = _initialize_replay_generated_tokens(
        rt,
        _replay_generated_tokens_tensor(max_new_tokens) if stop_on_eos
        else tensors.token_select.generated_tokens,
        max_new_tokens,
    )
    replay_generated_length = _replay_generated_length_tensor()
    replay_stopped = _replay_stopped_tensor()
    _initialize_replay_decode_control(rt, replay_generated_length, replay_stopped)
    replay_token_index = _replay_token_index_tensor()
    rt.register_inputs({replay_token_index: np.array([0], dtype=np.int64)})
    _run_qwen3_asr_token_store(
        rt,
        next_token=tensors.token_select.next_token,
        token_index=replay_token_index,
        done=tensors.token_select.done,
        generated_tokens=replay_generated,
        generated_length=replay_generated_length,
        stopped=replay_stopped,
        stop_on_eos=stop_on_eos,
    )
    if stop_on_eos and _replay_decode_stopped(rt, replay_stopped):
        return _finalize_replay_generated_tokens(
            rt, tensors, replay_generated, replay_generated_length,
        )

    if max_new_tokens == 1:
        if stop_on_eos:
            return _finalize_replay_generated_tokens(
                rt, tensors, replay_generated, replay_generated_length,
            )
        return replay_generated

    # Phase 2: Reuse a cached decode plan when this session has already recorded one.
    _ensure_kv_cache_capacity(rt, tensors, required_length=prompt_length + max_new_tokens)
    _register_decode_rope(
        rt, decode,
        cache_position=prompt_length,
        head_dim=head_dim,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
    )

    tensors_by_name = _replay_decode_tensors_by_name(
        tensors=tensors,
        replay_token_index=replay_token_index,
        replay_generated=replay_generated,
        replay_generated_length=replay_generated_length,
        replay_stopped=replay_stopped,
    )
    cached_plan = None
    if mode != "force_record":
        cached_plan = _find_cached_qwen3_asr_decode_replay_plan(
            rt,
            tensors_by_name=tensors_by_name,
            stop_on_eos=stop_on_eos,
        )
    if cached_plan is not None:
        _run_qwen3_asr_decode_replay_steps(
            rt,
            plan=cached_plan,
            decode=decode,
            replay_token_index=replay_token_index,
            prompt_length=prompt_length,
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            start_step=0,
            max_new_tokens=max_new_tokens,
            replay_stopped=replay_stopped,
            stop_on_eos=stop_on_eos,
        )
        if stop_on_eos:
            return _finalize_replay_generated_tokens(
                rt, tensors, replay_generated, replay_generated_length,
            )
        return replay_generated
    if mode == "require_cache":
        namespace = _qwen3_asr_decode_replay_namespace(stop_on_eos)
        raise RuntimeError(
            f"Qwen3-ASR decode replay cache miss for namespace {namespace!r}"
        )

    # Phase 3: First decode step (eager warm-up — materializes weights and creates pipelines)
    next_token_array = rt.read_request_state(tensors.token_select.next_token)
    rt.register_inputs({decode.input_ids: next_token_array.reshape(1, 1)})
    rt.register_inputs({replay_token_index: np.array([1], dtype=np.int64)})
    dispatch_start = len(rt.dispatch_records)
    run_qwen3_asr_text_decode(rt, decode, step=0, pytorch_compare=False)
    run_qwen3_asr_token_select(rt, tensors.token_select, logits=decode.logits)
    _run_qwen3_asr_token_store(
        rt,
        next_token=tensors.token_select.next_token,
        token_index=replay_token_index,
        done=tensors.token_select.done,
        generated_tokens=replay_generated,
        generated_length=replay_generated_length,
        stopped=replay_stopped,
        stop_on_eos=stop_on_eos,
    )
    dispatch_end = len(rt.dispatch_records)
    if stop_on_eos and _replay_decode_stopped(rt, replay_stopped):
        return _finalize_replay_generated_tokens(
            rt, tensors, replay_generated, replay_generated_length,
        )

    if max_new_tokens == 2:
        if stop_on_eos:
            return _finalize_replay_generated_tokens(
                rt, tensors, replay_generated, replay_generated_length,
            )
        return replay_generated

    # Phase 4: Build replay plan from the warm-up dispatches
    warmup_records = rt.dispatch_records[dispatch_start:dispatch_end]

    variant_by_shader: dict[str, ShaderVariant] = {}
    _collect_decode_variants(variant_by_shader)
    _collect_token_select_variants(variant_by_shader)
    _collect_token_store_variants(variant_by_shader, stop_on_eos=stop_on_eos)

    variants = [variant_by_shader[r.shader] for r in warmup_records]

    plan = rt.build_replay_plan(
        name="qwen3_asr_decode_step",
        frame_dispatch_records=list(warmup_records),
        variants=variants,
        tensors_by_name=tensors_by_name,
        dynamic_symbol_names=("S",),
        token_feedback_source=tensors.token_select.next_token,
        token_feedback_target=decode.input_ids,
    )
    if plan.readback_slots:
        plan.close()
        raise RuntimeError("Replay decode must not use readback slots")

    try:
        _run_qwen3_asr_decode_replay_steps(
            rt,
            plan=plan,
            decode=decode,
            replay_token_index=replay_token_index,
            prompt_length=prompt_length,
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            start_step=1,
            max_new_tokens=max_new_tokens,
            replay_stopped=replay_stopped,
            stop_on_eos=stop_on_eos,
        )
    except Exception:
        plan.close()
        raise
    rt.cache_replay_plan(_qwen3_asr_decode_replay_namespace(stop_on_eos), plan)

    if stop_on_eos:
        return _finalize_replay_generated_tokens(
            rt, tensors, replay_generated, replay_generated_length,
        )
    return replay_generated


def _finalize_generated_tokens(
    rt: RuntimeSession,
    tensors: Qwen3AsrTextTensors,
    tokens: list[int],
) -> LogicalTensor:
    """Write collected tokens into the generated_tokens tensor."""
    generated = tensors.token_select.generated_tokens
    array = np.array([tokens], dtype=np.int64)
    rt.grow_request_state(generated, array.shape)
    rt.initialize_request_state({generated: array})
    return generated


def _initialize_replay_generated_tokens(
    rt: RuntimeSession,
    generated: LogicalTensor,
    max_new_tokens: int,
) -> LogicalTensor:
    array = np.zeros((1, max_new_tokens), dtype=np.int64)
    rt.grow_request_state(generated, array.shape)
    rt.initialize_request_state({generated: array})
    return generated


def _finalize_replay_generated_tokens(
    rt: RuntimeSession,
    tensors: Qwen3AsrTextTensors,
    generated: LogicalTensor,
    generated_length: LogicalTensor,
) -> LogicalTensor:
    length = int(rt.read_request_state(generated_length).flatten()[0])
    tokens = [int(token) for token in rt.read_request_state(generated).flatten()[:length]]
    return _finalize_generated_tokens(rt, tensors, tokens)


def _replay_decode_stopped(rt: RuntimeSession, stopped: LogicalTensor) -> bool:
    return bool(np.asarray(rt.read_request_state(stopped)).reshape(-1)[0])


def _initialize_replay_decode_control(
    rt: RuntimeSession,
    generated_length: LogicalTensor,
    stopped: LogicalTensor,
) -> None:
    rt.initialize_request_state({
        generated_length: np.array([0], dtype=np.uint32),
        stopped: np.array([0], dtype=np.uint32),
    })


def _replay_decode_tensors_by_name(
    *,
    tensors: Qwen3AsrTextTensors,
    replay_token_index: LogicalTensor,
    replay_generated: LogicalTensor,
    replay_generated_length: LogicalTensor,
    replay_stopped: LogicalTensor,
) -> dict[str, LogicalTensor]:
    tensors_by_name: dict[str, LogicalTensor] = {}
    _collect_all_tensors(tensors, tensors_by_name)
    tensors_by_name[replay_token_index.name] = replay_token_index
    tensors_by_name[replay_generated.name] = replay_generated
    tensors_by_name[replay_generated_length.name] = replay_generated_length
    tensors_by_name[replay_stopped.name] = replay_stopped
    return tensors_by_name


def _find_cached_qwen3_asr_decode_replay_plan(
    rt: RuntimeSession,
    *,
    tensors_by_name: Mapping[str, LogicalTensor],
    stop_on_eos: bool,
) -> "ReplayPlan | None":
    namespace = _qwen3_asr_decode_replay_namespace(stop_on_eos)
    for plan in rt.cached_replay_plans(namespace):
        try:
            rt.rebind_replay_plan(plan, tensors_by_name=tensors_by_name)
        except (KeyError, ValueError):
            continue
        return plan
    return None


def _run_qwen3_asr_decode_replay_steps(
    rt: RuntimeSession,
    *,
    plan: "ReplayPlan",
    decode: Qwen3AsrTextDecodeTensors,
    replay_token_index: LogicalTensor,
    prompt_length: int,
    head_dim: int,
    rope_theta: float,
    mrope_section: tuple[int, ...],
    start_step: int,
    max_new_tokens: int,
    replay_stopped: LogicalTensor,
    stop_on_eos: bool,
) -> None:
    from torch2vk.runtime.replay import execute_replay

    for step in range(start_step, max_new_tokens - 1):
        cache_pos = prompt_length + step
        pos_3d = np.full((3, 1, 1), cache_pos, dtype=np.int64)
        cos_data, sin_data = precompute_qwen3_asr_mrope(
            position_ids=pos_3d,
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )
        _write_to_tensor_buffer(
            rt,
            decode.cache_position,
            np.array([cache_pos], dtype=np.int64),
        )
        _write_to_tensor_buffer(rt, decode.rope_cos, cos_data)
        _write_to_tensor_buffer(rt, decode.rope_sin, sin_data)
        _write_to_tensor_buffer(
            rt,
            replay_token_index,
            np.array([step + 1], dtype=np.int64),
        )
        execute_replay(plan, dynamic_symbols=_qwen3_asr_decode_dynamic_symbols(decode))
        if stop_on_eos and _replay_decode_stopped(rt, replay_stopped):
            break


def _qwen3_asr_decode_replay_namespace(stop_on_eos: bool) -> str:
    return f"qwen3_asr_decode_step:stop_on_eos={int(stop_on_eos)}"


def _qwen3_asr_decode_dynamic_symbols(
    decode: Qwen3AsrTextDecodeTensors,
) -> dict[str, int]:
    if not decode.layers:
        raise ValueError("Qwen3-ASR decode replay requires at least one decoder layer")
    return {"S": decode.layers[0].key_cache.concrete_shape[2]}


def _replay_generated_tokens_tensor(max_new_tokens: int) -> LogicalTensor:
    return LogicalTensor(
        name="qwen3_asr.replay.generated_tokens",
        spec=TensorSpec(dtype="int64", shape=(1, max_new_tokens)),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
        semantic=TensorSemantic.TOKEN,
    )


def _replay_generated_length_tensor() -> LogicalTensor:
    return LogicalTensor(
        name="qwen3_asr.replay.generated_length",
        spec=TensorSpec(dtype="uint32", shape=(1,)),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
        semantic=TensorSemantic.TOKEN,
    )


def _replay_stopped_tensor() -> LogicalTensor:
    return LogicalTensor(
        name="qwen3_asr.replay.stopped",
        spec=TensorSpec(dtype="uint32", shape=(1,)),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
        semantic=TensorSemantic.TOKEN,
    )


def _replay_token_index_tensor() -> LogicalTensor:
    return LogicalTensor(
        name="qwen3_asr.replay.token_index",
        spec=TensorSpec(dtype="int64", shape=(1,)),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )


def _run_qwen3_asr_token_store(
    rt: RuntimeSession,
    *,
    next_token: LogicalTensor,
    token_index: LogicalTensor,
    done: LogicalTensor,
    generated_tokens: LogicalTensor,
    generated_length: LogicalTensor,
    stopped: LogicalTensor,
    stop_on_eos: bool,
) -> None:
    from models.qwen3_asr.shaders.token_store_f32 import (
        QWEN3_ASR_TOKEN_STORE_EOS_F32,
        QWEN3_ASR_TOKEN_STORE_F32,
    )

    variant = QWEN3_ASR_TOKEN_STORE_EOS_F32 if stop_on_eos else QWEN3_ASR_TOKEN_STORE_F32

    with rt.frame("qwen3_asr.token_store"):
        variant(
            rt,
            next_token=next_token,
            token_index=token_index,
            done=done,
            generated_tokens=generated_tokens,
            generated_length=generated_length,
            stopped=stopped,
        )


def _write_to_tensor_buffer(
    rt: RuntimeSession,
    tensor: LogicalTensor,
    data: np.ndarray,
) -> None:
    """Write numpy data into a materialized tensor's host-visible buffer."""
    contiguous = np.ascontiguousarray(data, dtype=tensor.spec.dtype)
    if tensor.buffer is None:
        raise RuntimeError(f"Tensor {tensor.name} not materialized for replay write")
    raw_bytes = memoryview(contiguous).cast("B")
    if raw_bytes.nbytes > tensor.buffer.nbytes:
        raise ValueError(
            f"{tensor.name} replay write has {raw_bytes.nbytes} bytes, "
            f"buffer only has {tensor.buffer.nbytes}"
        )
    tensor.buffer.allocation.buffer.write_bytes_at(
        tensor.buffer.offset, raw_bytes
    )
    rt.device.memory_manager.host_upload_ring.flush(
        allocation=tensor.buffer.allocation,
        size=raw_bytes.nbytes,
    )


def _collect_decode_variants(out: dict[str, "ShaderVariant"]) -> None:
    from models.qwen3_asr.shaders.text_add_3d_f32 import QWEN3_ASR_TEXT_ADD_3D_F32
    from models.qwen3_asr.shaders.text_attention_decode_f32 import QWEN3_ASR_TEXT_ATTENTION_DECODE_F32
    from models.qwen3_asr.shaders.text_embed_lookup_f32 import QWEN3_ASR_TEXT_EMBED_LOOKUP_F32
    from models.qwen3_asr.shaders.text_kv_cache_write_f32 import QWEN3_ASR_TEXT_KV_CACHE_WRITE_DECODE_F32
    from models.qwen3_asr.shaders.text_linear_nobias_f32 import QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32
    from models.qwen3_asr.shaders.text_qk_norm_f32 import QWEN3_ASR_TEXT_QK_NORM_F32
    from models.qwen3_asr.shaders.text_rms_norm_f32 import QWEN3_ASR_TEXT_RMS_NORM_F32
    from models.qwen3_asr.shaders.text_rope_f32 import QWEN3_ASR_TEXT_ROPE_F32
    from models.qwen3_asr.shaders.text_swiglu_f32 import QWEN3_ASR_TEXT_SWIGLU_F32

    for v in (
        QWEN3_ASR_TEXT_ADD_3D_F32,
        QWEN3_ASR_TEXT_ATTENTION_DECODE_F32,
        QWEN3_ASR_TEXT_EMBED_LOOKUP_F32,
        QWEN3_ASR_TEXT_KV_CACHE_WRITE_DECODE_F32,
        QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32,
        QWEN3_ASR_TEXT_QK_NORM_F32,
        QWEN3_ASR_TEXT_RMS_NORM_F32,
        QWEN3_ASR_TEXT_ROPE_F32,
        QWEN3_ASR_TEXT_SWIGLU_F32,
    ):
        out[v.name] = v


def _collect_token_select_variants(out: dict[str, "ShaderVariant"]) -> None:
    from models.qwen3_asr.shaders.token_select_f32 import QWEN3_ASR_TOKEN_SELECT_GREEDY_F32
    out[QWEN3_ASR_TOKEN_SELECT_GREEDY_F32.name] = QWEN3_ASR_TOKEN_SELECT_GREEDY_F32


def _collect_token_store_variants(
    out: dict[str, "ShaderVariant"],
    *,
    stop_on_eos: bool,
) -> None:
    from models.qwen3_asr.shaders.token_store_f32 import (
        QWEN3_ASR_TOKEN_STORE_EOS_F32,
        QWEN3_ASR_TOKEN_STORE_F32,
    )

    variant = QWEN3_ASR_TOKEN_STORE_EOS_F32 if stop_on_eos else QWEN3_ASR_TOKEN_STORE_F32
    out[variant.name] = variant


def _collect_all_tensors(
    tensors: Qwen3AsrTextTensors,
    out: dict[str, LogicalTensor],
) -> None:
    """Collect all LogicalTensors from the text tensors struct by name."""
    decode = tensors.decode
    ts = tensors.token_select

    _collect_decode_tensor_fields(decode, out)

    for layer in decode.layers:
        _collect_layer_tensor_fields(layer, out)

    _collect_token_select_tensor_fields(ts, out)


def _collect_decode_tensor_fields(
    decode: Qwen3AsrTextDecodeTensors,
    out: dict[str, LogicalTensor],
) -> None:
    for field in dataclass_fields(Qwen3AsrTextDecodeTensors):
        value = getattr(decode, field.name)
        if isinstance(value, LogicalTensor):
            out[value.name] = value


def _collect_layer_tensor_fields(
    layer: Qwen3AsrTextLayerTensors,
    out: dict[str, LogicalTensor],
) -> None:
    for field in dataclass_fields(Qwen3AsrTextLayerTensors):
        value = getattr(layer, field.name)
        if isinstance(value, LogicalTensor):
            out[value.name] = value


def _collect_token_select_tensor_fields(
    token_select: Qwen3AsrTokenSelectTensors,
    out: dict[str, LogicalTensor],
) -> None:
    for field in dataclass_fields(Qwen3AsrTokenSelectTensors):
        value = getattr(token_select, field.name)
        if isinstance(value, LogicalTensor):
            out[value.name] = value
