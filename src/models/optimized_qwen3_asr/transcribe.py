"""End-to-end Qwen3-ASR transcription API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from models.hf_cache import load_config_json, resolve_cached_model
from models.optimized_qwen3_asr.audio_tower import run_qwen3_asr_audio_tower
from models.optimized_qwen3_asr.execution import (
    QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS,
    Qwen3AsrPreparedInputs,
    Qwen3AsrProcessorLike,
    prepare_qwen3_asr_inputs,
    run_qwen3_asr_greedy_decode_loop,
    run_qwen3_asr_replay_decode_loop,
)
from models.optimized_qwen3_asr.pytorch.example import REPO_ID
from models.optimized_qwen3_asr.tensors.audio_tower import declare_qwen3_asr_audio_tower_tensors
from models.optimized_qwen3_asr.tensors.text import (
    Qwen3AsrTextTensors,
    declare_qwen3_asr_text_tensors,
)
from torch2vk.runtime.compare import CompareAssertionError, TensorCompareResult
from torch2vk.runtime.session import RuntimeSession

_MAX_NEW_TOKENS = 256

JsonObject = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Qwen3AsrTranscription:
    text: str
    tokens: tuple[int, ...]


class Qwen3AsrDebugError(RuntimeError):
    """Raised when diagnostic PyTorch comparison finds a runtime mismatch."""

    def __init__(self, message: str, *, result: TensorCompareResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True, slots=True)
class _Qwen3AsrRuntimeConfig:
    audio_encoder_layers: int
    rope_theta: float
    mrope_section: tuple[int, ...]
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    decoder_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int


class Qwen3AsrRecognizer:
    """Reusable Vulkan-backed Qwen3-ASR recognizer."""

    def __init__(
        self,
        *,
        rt: RuntimeSession,
        model_dir: Path,
        config: _Qwen3AsrRuntimeConfig,
        pytorch_compare: bool,
    ) -> None:
        self._rt = rt
        self._model_dir = model_dir
        self._config = config
        self._pytorch_compare = pytorch_compare
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        device_index: int = 0,
        artifact_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        profile_dir: str | Path | None = None,
        pytorch_compare: bool = False,
    ) -> "Qwen3AsrRecognizer":
        resolved_model_dir = resolve_cached_model(REPO_ID, model_dir)
        config = _load_runtime_config(resolved_model_dir)
        rt = RuntimeSession.open(
            device_index=device_index,
            artifact_dir=artifact_dir,
            model_dir=resolved_model_dir,
            profile_dir=profile_dir,
        )
        return cls(
            rt=rt,
            model_dir=resolved_model_dir,
            config=config,
            pytorch_compare=pytorch_compare,
        )

    def __enter__(self) -> "Qwen3AsrRecognizer":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._rt.close()
            self._closed = True

    def transcribe(
        self,
        wav: str | Path | np.ndarray,
        *,
        language: str | None = "English",
        max_new_tokens: int = _MAX_NEW_TOKENS,
        pytorch_compare: bool | None = None,
    ) -> str:
        return self.transcribe_with_tokens(
            wav,
            language=language,
            max_new_tokens=max_new_tokens,
            pytorch_compare=pytorch_compare,
        ).text

    def transcribe_with_tokens(
        self,
        wav: str | Path | np.ndarray,
        *,
        language: str | None = "English",
        max_new_tokens: int = _MAX_NEW_TOKENS,
        pytorch_compare: bool | None = None,
    ) -> Qwen3AsrTranscription:
        if self._closed:
            raise RuntimeError("Qwen3AsrRecognizer is closed")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        compare = self._pytorch_compare if pytorch_compare is None else pytorch_compare

        try:
            processor, text_tensors = self._run_audio_and_prepare_text(
                wav=wav,
                language=language,
                max_new_tokens=max_new_tokens,
                pytorch_compare=compare,
            )
            if compare:
                generated = run_qwen3_asr_greedy_decode_loop(
                    self._rt,
                    text_tensors,
                    max_new_tokens=max_new_tokens,
                    rope_theta=self._config.rope_theta,
                    mrope_section=self._config.mrope_section,
                    pytorch_compare=True,
                    stop_on_eos=True,
                )
            else:
                generated = run_qwen3_asr_replay_decode_loop(
                    self._rt,
                    text_tensors,
                    max_new_tokens=max_new_tokens,
                    rope_theta=self._config.rope_theta,
                    mrope_section=self._config.mrope_section,
                    stop_on_eos=True,
                )
        except CompareAssertionError as exc:
            raise Qwen3AsrDebugError(
                _format_qwen3_asr_compare_failure(exc, artifact_dir=self._rt.artifact_dir),
                result=exc.result,
            ) from exc
        tokens = tuple(int(token) for token in self._rt.read_request_state(generated).flatten())
        return Qwen3AsrTranscription(
            text=_decode_generated_text(processor, tokens),
            tokens=tokens,
        )

    def _run_audio_and_prepare_text(
        self,
        *,
        wav: str | Path | np.ndarray,
        language: str | None,
        max_new_tokens: int,
        pytorch_compare: bool,
    ) -> tuple[Qwen3AsrProcessorLike, Qwen3AsrTextTensors]:
        processor, prepared = prepare_qwen3_asr_inputs(
            model_dir=self._model_dir,
            wav=wav,
            language=language,
        )
        audio_feature_len = prepared.audio_feature_length
        input_features = np.ascontiguousarray(
            prepared.input_features[0, :, :audio_feature_len],
            dtype=np.float32,
        )
        feature_lens = np.array([audio_feature_len], dtype=np.int64)

        audio_tensors = declare_qwen3_asr_audio_tower_tensors(
            input_features_shape=input_features.shape,
            encoder_layers=self._config.audio_encoder_layers,
        )
        text_tensors = declare_qwen3_asr_text_tensors(
            prompt_length=prepared.prompt_length,
            audio_tokens=audio_tensors.last_hidden_state.concrete_shape[0],
            max_sequence_length=prepared.prompt_length + max_new_tokens,
            hidden_size=self._config.hidden_size,
            intermediate_size=self._config.intermediate_size,
            vocab_size=self._config.vocab_size,
            decoder_layers=self._config.decoder_layers,
            num_attention_heads=self._config.num_attention_heads,
            num_key_value_heads=self._config.num_key_value_heads,
            head_dim=self._config.head_dim,
            audio_features=audio_tensors.last_hidden_state,
            pytorch_input_features_shape=prepared.input_features.shape,
            pytorch_feature_attention_mask_shape=prepared.feature_attention_mask.shape,
        )
        self._rt.set_model_tensors(text_tensors)

        self._rt.register_inputs({
            audio_tensors.input_features: input_features,
            audio_tensors.feature_lens: feature_lens,
        })
        run_qwen3_asr_audio_tower(
            self._rt,
            audio_tensors,
            pytorch_compare=pytorch_compare,
        )
        self._register_text_inputs(text_tensors, prepared=prepared)
        return processor, text_tensors

    def _register_text_inputs(
        self,
        text_tensors: Qwen3AsrTextTensors,
        *,
        prepared: Qwen3AsrPreparedInputs,
    ) -> None:
        prefill_input_features = text_tensors.prefill.input_features
        prefill_feature_attention_mask = text_tensors.prefill.feature_attention_mask
        if prefill_input_features is None or prefill_feature_attention_mask is None:
            raise RuntimeError("Qwen3-ASR text tensors are missing prefill audio inputs")
        self._rt.register_inputs({
            text_tensors.prefill.input_ids: prepared.input_ids,
            text_tensors.prefill.attention_mask: prepared.attention_mask,
            prefill_input_features: prepared.input_features,
            prefill_feature_attention_mask: prepared.feature_attention_mask,
            text_tensors.token_select.eos_token_ids: np.array(
                QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS,
                dtype=np.int64,
            ),
        })


def transcribe_wav(
    wav: str | Path | np.ndarray,
    *,
    language: str | None = "English",
    max_new_tokens: int = _MAX_NEW_TOKENS,
    device_index: int = 0,
    artifact_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    profile_dir: str | Path | None = None,
    pytorch_compare: bool = False,
) -> str:
    with Qwen3AsrRecognizer.open(
        device_index=device_index,
        artifact_dir=artifact_dir,
        model_dir=model_dir,
        profile_dir=profile_dir,
        pytorch_compare=pytorch_compare,
    ) as recognizer:
        return recognizer.transcribe(
            wav,
            language=language,
            max_new_tokens=max_new_tokens,
        )


def _decode_generated_text(
    processor: Qwen3AsrProcessorLike,
    tokens: tuple[int, ...],
) -> str:
    decoded = processor.batch_decode(
        np.array([tokens], dtype=np.int64),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if len(decoded) != 1:
        raise ValueError(f"Expected one decoded transcript, got {len(decoded)}")
    return decoded[0]


def _format_qwen3_asr_compare_failure(
    exc: CompareAssertionError,
    *,
    artifact_dir: Path,
) -> str:
    result = exc.result
    lines = [
        "Qwen3-ASR PyTorch comparison failed while transcribing.",
        "This usually means a Vulkan shader produced values that diverged from the PyTorch reference.",
        str(exc),
    ]
    drilldown = _load_drilldown(result.drilldown_artifact_path)
    if drilldown is not None:
        dispatch = drilldown.get("dispatch")
        if isinstance(dispatch, Mapping):
            shader = dispatch.get("shader")
            frame = dispatch.get("frame")
            index = dispatch.get("index")
            lines.append(
                "  suspect_dispatch: "
                f"frame={frame}, shader={shader}, dispatch_index={index}"
            )
            if isinstance(shader, str):
                glsl_paths = _generated_glsl_paths(artifact_dir, shader)
                if glsl_paths:
                    lines.append("  generated_glsl:")
                    lines.extend(f"    {path}" for path in glsl_paths[:3])
        classification = drilldown.get("classification")
        if isinstance(classification, str):
            lines.append(f"  drilldown_classification: {classification}")
    return "\n".join(lines)


def _load_drilldown(path: str | None) -> Mapping[str, object] | None:
    if path is None:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _generated_glsl_paths(artifact_dir: Path, shader: str) -> tuple[str, ...]:
    paths = sorted(
        artifact_dir.glob(f"{shader}.*.comp"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return tuple(str(path) for path in paths)


def _load_runtime_config(model_dir: Path) -> _Qwen3AsrRuntimeConfig:
    config = _expect_mapping(load_config_json(model_dir), "config")
    thinker_config = _expect_mapping(config["thinker_config"], "thinker_config")
    audio_config = _expect_mapping(thinker_config["audio_config"], "audio_config")
    text_config = _expect_mapping(thinker_config["text_config"], "text_config")
    return _Qwen3AsrRuntimeConfig(
        audio_encoder_layers=_config_int(audio_config, "encoder_layers"),
        rope_theta=_config_float(text_config, "rope_theta", 5_000_000.0),
        mrope_section=_mrope_section(text_config),
        hidden_size=_config_int(text_config, "hidden_size"),
        intermediate_size=_config_int(text_config, "intermediate_size"),
        vocab_size=_config_int(text_config, "vocab_size"),
        decoder_layers=_config_int(text_config, "num_hidden_layers"),
        num_attention_heads=_config_int(text_config, "num_attention_heads"),
        num_key_value_heads=_config_int(text_config, "num_key_value_heads"),
        head_dim=_config_int(text_config, "head_dim"),
    )


def _expect_mapping(value: object, name: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _config_int(config: JsonObject, key: str) -> int:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an int, got {type(value).__name__}")
    return value


def _config_float(config: JsonObject, key: str, default: float) -> float:
    value = config.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{key} must be numeric, got {type(value).__name__}")
    return float(value)


def _mrope_section(text_config: JsonObject) -> tuple[int, ...]:
    rope_scaling = text_config.get("rope_scaling")
    if not isinstance(rope_scaling, Mapping):
        return (24, 20, 20)
    section = rope_scaling.get("mrope_section", (24, 20, 20))
    if not isinstance(section, Sequence) or isinstance(section, str | bytes):
        raise TypeError("rope_scaling.mrope_section must be a sequence")
    return tuple(_expect_int(item, "rope_scaling.mrope_section") for item in section)


def _expect_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value
