"""Qwen3-ASR runtime orchestration entry points."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeGuard

import numpy as np
import numpy.typing as npt

from models.hf_cache import resolve_cached_model
from models.optimized_qwen3_asr.audio_tower import (
    run_qwen3_asr_audio_tower as run_qwen3_asr_audio_tower,
)
from models.optimized_qwen3_asr.pytorch.example import REPO_ID
from models.optimized_qwen3_asr.text_decode import run_qwen3_asr_text_decode
from models.optimized_qwen3_asr.text_prefill import run_qwen3_asr_text_prefill
from models.optimized_qwen3_asr.tensors.text import (
    Qwen3AsrTextDecodeTensors,
    Qwen3AsrTextPrefillTensors,
    Qwen3AsrTextTensors,
)
from models.optimized_qwen3_asr.token_select import run_qwen3_asr_token_select
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.rope_table import (
    declare_rope_start_position_tensor,
    declare_rope_theta_tensor,
    run_rope_table_f32,
)
from torch2vk.runtime.session import RuntimeSession

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
    generated_length = tensors.token_select.generated_length
    stopped = tensors.token_select.stopped
    token_index = tensors.token_select.token_index
    _initialize_replay_decode_control(rt, generated_length, stopped)
    _append_generated_token(
        rt,
        tensors,
        token_index_tensor=token_index,
        token_index=0,
        generated_length=generated_length,
        stopped=stopped,
        stop_on_eos=stop_on_eos,
    )
    if stop_on_eos and _token_select_done(rt, tensors):
        return tensors.token_select.generated_tokens

    prompt_length = prefill.input_ids.concrete_shape[-1]
    decode = tensors.decode

    baseline_stats = rt.device.allocation_stats()
    print(f"  [memory] Decode baseline: device_local={baseline_stats.device_local_live_bytes / 1024**2:.1f} MB, "
          f"reserved={baseline_stats.device_local_reserved_bytes / 1024**2:.1f} MB")
    memory_trace: list[tuple[int, float, float]] = []

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
        if pytorch_compare:
            run_qwen3_asr_text_decode(rt, decode, step=step, pytorch_compare=True)
            run_qwen3_asr_token_select(rt, tensors.token_select, logits=decode.logits)
        else:
            run_qwen3_asr_text_decode(
                rt,
                decode,
                step=step,
                pytorch_compare=False,
            )
            run_qwen3_asr_token_select(rt, tensors.token_select, logits=decode.logits)
        _append_generated_token(
            rt,
            tensors,
            token_index_tensor=token_index,
            token_index=step + 1,
            generated_length=generated_length,
            stopped=stopped,
            stop_on_eos=stop_on_eos,
        )

        stats = rt.device.allocation_stats()
        memory_trace.append((step, stats.device_local_live_bytes / 1024**2, stats.device_local_reserved_bytes / 1024**2))
        if step < 5 or step % 20 == 0:
            print(f"  [memory] Step {step}: gpu={stats.device_local_live_bytes / 1024**2:.1f}MB")

        if stop_on_eos and _token_select_done(rt, tensors):
            break

    final_stats = rt.device.allocation_stats()
    print("\n  [memory] === GPU Memory Trace ===")
    print(f"  [memory] Peak device_local live: {final_stats.device_local_peak_live_bytes / 1024**2:.1f} MB")
    print(f"  [memory] Peak device_local reserved: {final_stats.device_local_peak_reserved_bytes / 1024**2:.1f} MB")
    print(f"  [memory] Final device_local live: {final_stats.device_local_live_bytes / 1024**2:.1f} MB")
    print(f"  [memory] Steps sampled: {len(memory_trace)}")
    print("  [memory] Step | GPU Live (MB) | GPU Reserved (MB)")
    print("  [memory] -----|---------------|------------------")
    for s, live, reserved in memory_trace:
        print(f"  [memory] {s:4d} | {live:13.1f} | {reserved:17.1f}")

    return tensors.token_select.generated_tokens


def _register_prefill_rope(
    rt: RuntimeSession,
    prefill: Qwen3AsrTextPrefillTensors,
    *,
    rope_theta: float,
    mrope_section: tuple[int, ...],
) -> None:
    _require_supported_mrope_section(mrope_section)
    start_position = declare_rope_start_position_tensor("qwen3_asr.rope.start_position")
    rope_theta_tensor = declare_rope_theta_tensor("qwen3_asr.rope.theta")
    rt.register_inputs(
        {
            start_position: np.array([0], dtype=np.int64),
            rope_theta_tensor: np.array([rope_theta], dtype=np.float32),
        }
    )
    run_rope_table_f32(
        rt,
        start_position=start_position,
        theta=rope_theta_tensor,
        cos=prefill.rope_cos,
        sin=prefill.rope_sin,
        frame_name="qwen3_asr.rope_table",
    )


def _register_decode_rope(
    rt: RuntimeSession,
    decode: Qwen3AsrTextDecodeTensors,
    *,
    cache_position: int,
    head_dim: int,
    rope_theta: float,
    mrope_section: tuple[int, ...],
) -> None:
    del head_dim
    _require_supported_mrope_section(mrope_section)
    rope_theta_tensor = declare_rope_theta_tensor("qwen3_asr.rope.theta")
    rt.register_inputs(
        {
            decode.cache_position: np.array([cache_position], dtype=np.int64),
            rope_theta_tensor: np.array([rope_theta], dtype=np.float32),
        }
    )
    run_rope_table_f32(
        rt,
        start_position=decode.cache_position,
        theta=rope_theta_tensor,
        cos=decode.rope_cos,
        sin=decode.rope_sin,
        frame_name="qwen3_asr.rope_table",
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
    *,
    token_index_tensor: LogicalTensor,
    token_index: int,
    generated_length: LogicalTensor,
    stopped: LogicalTensor,
    stop_on_eos: bool,
) -> None:
    generated = tensors.token_select.generated_tokens
    batch = generated.concrete_shape[0]
    rt.grow_request_state(generated, (batch, token_index + 1))
    rt.register_inputs({token_index_tensor: np.array([token_index], dtype=np.int64)})
    _run_qwen3_asr_token_store(
        rt,
        next_token=tensors.token_select.next_token,
        token_index=token_index_tensor,
        done=tensors.token_select.done,
        generated_tokens=generated,
        generated_length=generated_length,
        stopped=stopped,
        stop_on_eos=stop_on_eos,
    )


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
        tensors.token_select.replay_generated_tokens if stop_on_eos
        else tensors.token_select.generated_tokens,
        max_new_tokens,
    )
    replay_generated_length = tensors.token_select.generated_length
    replay_stopped = tensors.token_select.stopped
    _initialize_replay_decode_control(rt, replay_generated_length, replay_stopped)
    replay_token_index = tensors.token_select.token_index
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

    cached_plan = None
    if mode != "force_record":
        cached_plan = _find_cached_qwen3_asr_decode_replay_plan(
            rt,
            stop_on_eos=stop_on_eos,
        )
    if cached_plan is not None:
        _run_qwen3_asr_decode_replay_steps(
            rt,
            plan=cached_plan,
            decode=decode,
            next_token=tensors.token_select.next_token,
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
    decode_step_frame = "qwen3_asr.decode_step.0000"
    with rt.frame(decode_step_frame):
        run_qwen3_asr_text_decode(
            rt,
            decode,
            step=0,
            pytorch_compare=False,
        )
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

    plan = rt.build_replay_plan(
        name="qwen3_asr_decode_step",
        frame=decode_step_frame,
    )
    if plan.readback_slots:
        plan.close()
        raise RuntimeError("Replay decode must not use readback slots")

    try:
        _run_qwen3_asr_decode_replay_steps(
            rt,
            plan=plan,
            decode=decode,
            next_token=tensors.token_select.next_token,
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


def _find_cached_qwen3_asr_decode_replay_plan(
    rt: RuntimeSession,
    *,
    stop_on_eos: bool,
) -> "ReplayPlan | None":
    namespace = _qwen3_asr_decode_replay_namespace(stop_on_eos)
    for plan in rt.cached_replay_plans(namespace):
        if rt.replay_plan_compatible(plan):
            return plan
    return None


def _run_qwen3_asr_decode_replay_steps(
    rt: RuntimeSession,
    *,
    plan: "ReplayPlan",
    decode: Qwen3AsrTextDecodeTensors,
    next_token: LogicalTensor,
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
    from torch2vk.runtime.replay import execute_replay, stage_replay_step_inputs

    del head_dim
    _require_supported_mrope_section(mrope_section)
    rope_theta_tensor = declare_rope_theta_tensor("qwen3_asr.rope.theta")
    for step in range(start_step, max_new_tokens - 1):
        cache_pos = prompt_length + step
        next_token_value = rt.read_request_state(next_token)
        token_index_value = np.array([step + 1], dtype=np.int64)
        step_inputs = {
            decode.input_ids: next_token_value.reshape(1, 1),
            decode.cache_position: np.array([cache_pos], dtype=np.int64),
            rope_theta_tensor: np.array([rope_theta], dtype=np.float32),
            replay_token_index: token_index_value,
        }
        rt.register_inputs(step_inputs)
        run_rope_table_f32(
            rt,
            start_position=decode.cache_position,
            theta=rope_theta_tensor,
            cos=decode.rope_cos,
            sin=decode.rope_sin,
            frame_name="qwen3_asr.rope_table",
        )
        stage_replay_step_inputs(
            rt,
            plan=plan,
            inputs=step_inputs,
            write_through=(decode.input_ids, decode.cache_position, replay_token_index),
        )
        execute_replay(plan, dynamic_symbols=_qwen3_asr_decode_dynamic_symbols(decode))
        if stop_on_eos and _replay_decode_stopped(rt, replay_stopped):
            break


def _qwen3_asr_decode_replay_namespace(stop_on_eos: bool) -> str:
    return f"qwen3_asr_decode_step:v5:stop_on_eos={int(stop_on_eos)}"


def _qwen3_asr_decode_dynamic_symbols(
    decode: Qwen3AsrTextDecodeTensors,
) -> dict[str, int]:
    if not decode.layers:
        raise ValueError("Qwen3-ASR decode replay requires at least one decoder layer")
    return {"S": decode.layers[0].key_cache.concrete_shape[2]}


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
    from models.optimized_qwen3_asr.shaders.token_store_f32 import (
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


def _require_supported_mrope_section(mrope_section: tuple[int, ...]) -> None:
    if tuple(mrope_section) != (24, 20, 20):
        raise NotImplementedError(
            f"qwen3_asr rope_table shader currently supports mrope_section=(24, 20, 20), got {mrope_section}"
        )

