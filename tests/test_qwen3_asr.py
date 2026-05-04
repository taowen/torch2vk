"""Qwen3-ASR end-to-end runtime coverage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from models.hf_cache import load_config_json, resolve_cached_model
from models.qwen3_asr.audio_tower import run_qwen3_asr_audio_tower
from models.qwen3_asr.execution import (
    QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS,
    prepare_qwen3_asr_inputs,
    run_qwen3_asr_greedy_decode_loop,
    run_qwen3_asr_replay_decode_loop,
)
from models.qwen3_asr.pytorch.example import REPO_ID
from models.qwen3_asr.tensors.audio_tower import (
    Qwen3AsrAudioTowerTensors,
    declare_qwen3_asr_audio_tower_tensors,
)
from models.qwen3_asr.tensors.text import (
    Qwen3AsrTextTensors,
    declare_qwen3_asr_text_tensors,
)
from torch2vk.runtime.session import RuntimeSession

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_ASKNOT_WAV = _FIXTURE_DIR / "qwen3_asr_asknot.wav"
_ZH_WAV = _FIXTURE_DIR / "qwen3_asr_zh.wav"
_EXPECTED_ASKNOT_TEXT = (
    "And so, my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)
_EXPECTED_ZH_TEXT = "甚至出现交易几乎停滞的情况。"
_EXPECTED_ASKNOT_TOKEN_PREFIX = (3036, 773, 11, 847)
_MAX_NEW_TOKENS = 256
_REPLAY_NAMESPACE = "qwen3_asr_decode_step:stop_on_eos=1"

JsonObject = Mapping[str, object]
DecodeKind = Literal["greedy", "replay"]
ReplayMode = Literal["default", "require_cache", "force_record"]


@dataclass(frozen=True, slots=True)
class Qwen3AsrRequestResult:
    tokens: tuple[int, ...]
    text: str
    eos_reached: bool
    prompt_length: int
    audio_feature_length: int


def test_qwen3_asr_e2e_transcribes_fixture(tmp_path: Path) -> None:
    """Run fixture wav through Vulkan shaders and decode the full transcript."""
    resolved_model_dir, audio_config, text_config = _load_model_config()
    rope_theta = _config_float(text_config, "rope_theta", 5_000_000.0)
    mrope_section = _mrope_section(text_config)

    artifact_dir = tmp_path / "qwen3_asr_e2e"
    with RuntimeSession.open(
        device_index=0,
        artifact_dir=artifact_dir,
        model_dir=resolved_model_dir,
    ) as rt:
        result = _run_qwen3_asr_request(
            rt,
            model_dir=resolved_model_dir,
            audio_config=audio_config,
            text_config=text_config,
            wav=_ASKNOT_WAV,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            decode_kind="greedy",
            audio_pytorch_compare=True,
        )
        failed = [result for result in rt.compare_results if not result.passed]

    assert result.eos_reached
    assert result.tokens[:4] == _EXPECTED_ASKNOT_TOKEN_PREFIX
    assert result.text == _EXPECTED_ASKNOT_TEXT
    assert not failed, (
        f"{len(failed)} shader comparison(s) failed.\n"
        f"First failure: {failed[0].artifact_key}\n"
        f"Check drilldown artifacts in {artifact_dir}"
    )


def test_qwen3_asr_replay_decode_cache_reused_for_second_wav(tmp_path: Path) -> None:
    """A second compatible wav must use the cached decode replay plan."""
    resolved_model_dir, audio_config, text_config = _load_model_config()
    rope_theta = _config_float(text_config, "rope_theta", 5_000_000.0)
    mrope_section = _mrope_section(text_config)

    with RuntimeSession.open(
        device_index=0,
        artifact_dir=tmp_path / "qwen3_asr_replay",
        model_dir=resolved_model_dir,
    ) as rt:
        first = _run_qwen3_asr_request(
            rt,
            model_dir=resolved_model_dir,
            audio_config=audio_config,
            text_config=text_config,
            wav=_ASKNOT_WAV,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            decode_kind="replay",
            replay_mode="force_record",
        )
        assert first.eos_reached
        assert first.text == _EXPECTED_ASKNOT_TEXT

        cached_plans = rt.cached_replay_plans(_REPLAY_NAMESPACE)
        assert len(cached_plans) == 1
        command_buffer = cached_plans[0].command_buffer

        dispatch_start = len(rt.dispatch_records)
        second = _run_qwen3_asr_request(
            rt,
            model_dir=resolved_model_dir,
            audio_config=audio_config,
            text_config=text_config,
            wav=_ZH_WAV,
            language="Chinese",
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            decode_kind="replay",
            replay_mode="require_cache",
        )
        second_records = rt.dispatch_records[dispatch_start:]

        assert second.prompt_length != first.prompt_length
        assert second.audio_feature_length != first.audio_feature_length
        assert rt.cached_replay_plans(_REPLAY_NAMESPACE) == cached_plans
        assert cached_plans[0].command_buffer is command_buffer
        assert not any(
            record.frame.startswith("qwen3_asr.text_decode")
            for record in second_records
        )
        assert second.eos_reached
        assert len(second.tokens) < _MAX_NEW_TOKENS
        assert second.text == _EXPECTED_ZH_TEXT


def _run_qwen3_asr_request(
    rt: RuntimeSession,
    *,
    model_dir: Path,
    audio_config: JsonObject,
    text_config: JsonObject,
    wav: Path,
    rope_theta: float,
    mrope_section: tuple[int, ...],
    decode_kind: DecodeKind,
    language: str = "English",
    replay_mode: ReplayMode = "default",
    audio_pytorch_compare: bool = False,
    text_pytorch_compare: bool = False,
) -> Qwen3AsrRequestResult:
    processor, audio_tensors, text_tensors, prepared, input_features, feature_lens = (
        _prepare_qwen3_asr_request(
            model_dir=model_dir,
            audio_config=audio_config,
            text_config=text_config,
            wav=wav,
            language=language,
        )
    )

    rt.register_inputs({
        audio_tensors.input_features: input_features,
        audio_tensors.feature_lens: feature_lens,
    })
    run_qwen3_asr_audio_tower(
        rt,
        audio_tensors,
        pytorch_compare=audio_pytorch_compare,
    )
    _register_text_inputs(rt, text_tensors=text_tensors, prepared=prepared)

    if decode_kind == "greedy":
        generated = run_qwen3_asr_greedy_decode_loop(
            rt,
            text_tensors,
            max_new_tokens=_MAX_NEW_TOKENS,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            pytorch_compare=text_pytorch_compare,
            stop_on_eos=True,
        )
    elif decode_kind == "replay":
        generated = run_qwen3_asr_replay_decode_loop(
            rt,
            text_tensors,
            max_new_tokens=_MAX_NEW_TOKENS,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            stop_on_eos=True,
            mode=replay_mode,
        )
    else:
        raise ValueError(f"Unsupported decode_kind: {decode_kind!r}")

    tokens = tuple(int(token) for token in rt.read_request_state(generated).flatten())
    return Qwen3AsrRequestResult(
        tokens=tokens,
        text=_decode_generated_text(processor, tokens),
        eos_reached=any(token in QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS for token in tokens),
        prompt_length=prepared.prompt_length,
        audio_feature_length=prepared.audio_feature_length,
    )


def _prepare_qwen3_asr_request(
    *,
    model_dir: Path,
    audio_config: JsonObject,
    text_config: JsonObject,
    wav: Path,
    language: str,
) -> tuple[
    object,
    Qwen3AsrAudioTowerTensors,
    Qwen3AsrTextTensors,
    object,
    np.ndarray,
    np.ndarray,
]:
    processor, prepared = prepare_qwen3_asr_inputs(
        model_dir=model_dir,
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
        encoder_layers=_config_int(audio_config, "encoder_layers"),
    )
    text_tensors = declare_qwen3_asr_text_tensors(
        prompt_length=prepared.prompt_length,
        audio_tokens=audio_tensors.last_hidden_state.concrete_shape[0],
        max_sequence_length=prepared.prompt_length + _MAX_NEW_TOKENS,
        hidden_size=_config_int(text_config, "hidden_size"),
        intermediate_size=_config_int(text_config, "intermediate_size"),
        vocab_size=_config_int(text_config, "vocab_size"),
        decoder_layers=_config_int(text_config, "num_hidden_layers"),
        num_attention_heads=_config_int(text_config, "num_attention_heads"),
        num_key_value_heads=_config_int(text_config, "num_key_value_heads"),
        head_dim=_config_int(text_config, "head_dim"),
        audio_features=audio_tensors.last_hidden_state,
        pytorch_input_features_shape=prepared.input_features.shape,
        pytorch_feature_attention_mask_shape=prepared.feature_attention_mask.shape,
    )
    return processor, audio_tensors, text_tensors, prepared, input_features, feature_lens


def _decode_generated_text(processor: object, tokens: tuple[int, ...]) -> str:
    decoded = processor.batch_decode(
        np.array([tokens], dtype=np.int64),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if len(decoded) != 1:
        raise ValueError(f"Expected one decoded transcript, got {len(decoded)}")
    return decoded[0]


def _register_text_inputs(
    rt: RuntimeSession,
    *,
    text_tensors: Qwen3AsrTextTensors,
    prepared: object,
) -> None:
    prefill_input_features = text_tensors.prefill.input_features
    prefill_feature_attention_mask = text_tensors.prefill.feature_attention_mask
    assert prefill_input_features is not None
    assert prefill_feature_attention_mask is not None
    rt.register_inputs({
        text_tensors.prefill.input_ids: prepared.input_ids,
        text_tensors.prefill.attention_mask: prepared.attention_mask,
        prefill_input_features: prepared.input_features,
        prefill_feature_attention_mask: prepared.feature_attention_mask,
        text_tensors.token_select.eos_token_ids: np.array(
            QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS,
            dtype=np.int64,
        ),
    })


def _load_model_config() -> tuple[Path, JsonObject, JsonObject]:
    resolved_model_dir = resolve_cached_model(REPO_ID, None)
    config = _expect_mapping(load_config_json(resolved_model_dir), "config")
    thinker_config = _expect_mapping(config["thinker_config"], "thinker_config")
    audio_config = _expect_mapping(thinker_config["audio_config"], "audio_config")
    text_config = _expect_mapping(thinker_config["text_config"], "text_config")
    return resolved_model_dir, audio_config, text_config


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
