"""Test that replay decode produces the same tokens as eager decode."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from models.hf_cache import resolve_cached_model, load_config_json
from models.qwen3_asr.execution import (
    prepare_qwen3_asr_inputs,
    QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS,
)
from models.qwen3_asr.audio_tower import run_qwen3_asr_audio_tower
from models.qwen3_asr.tensors.audio_tower import declare_qwen3_asr_audio_tower_tensors
from models.qwen3_asr.tensors.text import declare_qwen3_asr_text_tensors
from models.qwen3_asr.pytorch.example import REPO_ID
from torch2vk.runtime.session import RuntimeSession

_FIXTURE_WAV = Path(__file__).resolve().parent / "fixtures" / "qwen3_asr_asknot.wav"


JsonObject = Mapping[str, object]


def _load_model_config() -> tuple[Path, JsonObject, JsonObject]:
    resolved_model_dir = resolve_cached_model(REPO_ID, None)
    config = _require_json_object(load_config_json(resolved_model_dir), "config")
    thinker_config = _require_json_object(config["thinker_config"], "thinker_config")
    audio_config = _require_json_object(thinker_config["audio_config"], "audio_config")
    text_config = _require_json_object(thinker_config["text_config"], "text_config")
    return resolved_model_dir, audio_config, text_config


def _require_json_object(value: object, name: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object, got {type(value).__name__}")
    return value


def _config_int(config: JsonObject, key: str) -> int:
    value = config[key]
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"{key} must be int-like, got {type(value).__name__}")


def _config_float(config: JsonObject, key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"{key} must be float-like, got {type(value).__name__}")


def _mrope_section(text_config: JsonObject) -> tuple[int, ...]:
    rope_scaling = text_config.get("rope_scaling")
    if not isinstance(rope_scaling, Mapping):
        return (24, 20, 20)
    section = rope_scaling.get("mrope_section", (24, 20, 20))
    if not isinstance(section, Sequence) or isinstance(section, str | bytes):
        raise TypeError("rope_scaling.mrope_section must be a sequence")
    return tuple(_int_like(item, "rope_scaling.mrope_section") for item in section)


def _int_like(value: object, name: str) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"{name} item must be int-like, got {type(value).__name__}")


@pytest.mark.parametrize("stop_on_eos", [False, True])
def test_replay_decode_loop_matches_eager(tmp_path: Path, *, stop_on_eos: bool) -> None:
    """Full replay decode loop produces same tokens as eager."""
    from models.qwen3_asr.execution import run_qwen3_asr_replay_decode_loop

    resolved_model_dir, audio_config, text_config = _load_model_config()
    _processor, prepared = prepare_qwen3_asr_inputs(
        model_dir=resolved_model_dir,
        wav=_FIXTURE_WAV,
        language="English",
    )

    encoder_layers = _config_int(audio_config, "encoder_layers")
    audio_feature_len = prepared.audio_feature_length
    input_features = np.ascontiguousarray(
        prepared.input_features[0, :, :audio_feature_len], dtype=np.float32,
    )
    feature_lens = np.array([audio_feature_len], dtype=np.int64)

    max_new_tokens = 4
    rope_theta = _config_float(text_config, "rope_theta", 5_000_000.0)
    mrope_section = _mrope_section(text_config)

    audio_tensors = declare_qwen3_asr_audio_tower_tensors(
        input_features_shape=input_features.shape,
        encoder_layers=encoder_layers,
    )
    text_tensors = declare_qwen3_asr_text_tensors(
        prompt_length=prepared.prompt_length,
        audio_tokens=audio_tensors.last_hidden_state.concrete_shape[0],
        max_sequence_length=prepared.prompt_length + max_new_tokens,
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

    artifact_dir = tmp_path / f"replay_loop_test_{stop_on_eos}"
    with RuntimeSession.open(
        device_index=0,
        artifact_dir=artifact_dir,
        model_dir=resolved_model_dir,
    ) as rt:
        rt.register_inputs({
            audio_tensors.input_features: input_features,
            audio_tensors.feature_lens: feature_lens,
        })
        run_qwen3_asr_audio_tower(rt, audio_tensors)

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
                QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS, dtype=np.int64,
            ),
        })

        generated_tensor = run_qwen3_asr_replay_decode_loop(
            rt, text_tensors,
            max_new_tokens=max_new_tokens,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            stop_on_eos=stop_on_eos,
        )
        replay_tokens = tuple(
            int(t) for t in rt.read_request_state(generated_tensor).flatten()
        )
        if stop_on_eos:
            dispatch_start = len(rt.dispatch_records)
            generated_tensor = run_qwen3_asr_replay_decode_loop(
                rt, text_tensors,
                max_new_tokens=max_new_tokens,
                rope_theta=rope_theta,
                mrope_section=mrope_section,
                stop_on_eos=stop_on_eos,
            )
            second_records = rt.dispatch_records[dispatch_start:]
            replay_tokens = tuple(
                int(t) for t in rt.read_request_state(generated_tensor).flatten()
            )
            assert not any(
                record.frame.startswith("qwen3_asr.text_decode")
                for record in second_records
            )

    assert replay_tokens == (3036, 773, 11, 847), f"Got {replay_tokens}"
