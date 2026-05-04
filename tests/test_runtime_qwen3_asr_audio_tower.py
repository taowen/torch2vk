"""End-to-end integration test: run qwen3_asr_asknot.wav through torch2vk compute shaders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from models.hf_cache import resolve_cached_model, load_config_json
from models.qwen3_asr.execution import (
    prepare_qwen3_asr_inputs,
    QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS,
)
from models.qwen3_asr.audio_tower import run_qwen3_asr_audio_tower
from models.qwen3_asr.execution import run_qwen3_asr_greedy_decode_loop
from models.qwen3_asr.tensors.audio_tower import declare_qwen3_asr_audio_tower_tensors
from models.qwen3_asr.tensors.text import declare_qwen3_asr_text_tensors
from models.qwen3_asr.pytorch.example import REPO_ID
from torch2vk.runtime.session import RuntimeSession

_FIXTURE_WAV = Path(__file__).resolve().parent / "fixtures" / "qwen3_asr_asknot.wav"


def test_qwen3_asr_e2e_with_pytorch_compare(tmp_path: Path) -> None:
    """Run fixture wav through Vulkan shaders with pytorch probe comparison."""
    resolved_model_dir = resolve_cached_model(REPO_ID, None)
    _processor, prepared = prepare_qwen3_asr_inputs(
        model_dir=resolved_model_dir,
        wav=_FIXTURE_WAV,
        language="English",
    )
    config = _expect_mapping(load_config_json(resolved_model_dir), "config")
    thinker_config = _expect_mapping(config["thinker_config"], "thinker_config")
    audio_config = _expect_mapping(thinker_config["audio_config"], "audio_config")
    text_config = _expect_mapping(thinker_config["text_config"], "text_config")
    encoder_layers = _expect_int(audio_config["encoder_layers"], "audio_config.encoder_layers")

    audio_feature_len = prepared.audio_feature_length
    input_features = np.ascontiguousarray(
        prepared.input_features[0, :, :audio_feature_len],
        dtype=np.float32,
    )
    feature_lens = np.array([audio_feature_len], dtype=np.int64)

    tensors = declare_qwen3_asr_audio_tower_tensors(
        input_features_shape=input_features.shape,
        encoder_layers=encoder_layers,
    )
    text_tensors = declare_qwen3_asr_text_tensors(
        prompt_length=prepared.prompt_length,
        audio_tokens=tensors.last_hidden_state.concrete_shape[0],
        max_sequence_length=prepared.prompt_length + 1,
        hidden_size=_expect_int(text_config["hidden_size"], "text_config.hidden_size"),
        intermediate_size=_expect_int(
            text_config["intermediate_size"], "text_config.intermediate_size"
        ),
        vocab_size=_expect_int(text_config["vocab_size"], "text_config.vocab_size"),
        decoder_layers=_expect_int(
            text_config["num_hidden_layers"], "text_config.num_hidden_layers"
        ),
        num_attention_heads=_expect_int(
            text_config["num_attention_heads"], "text_config.num_attention_heads"
        ),
        num_key_value_heads=_expect_int(
            text_config["num_key_value_heads"], "text_config.num_key_value_heads"
        ),
        head_dim=_expect_int(text_config["head_dim"], "text_config.head_dim"),
        audio_features=tensors.last_hidden_state,
    )

    max_new_tokens = 4
    rope_theta = _expect_float(text_config.get("rope_theta", 5_000_000.0), "text_config.rope_theta")
    rope_scaling = _optional_mapping(text_config.get("rope_scaling"))
    mrope_section = _int_tuple(rope_scaling.get("mrope_section"), default=(24, 20, 20))

    artifact_dir = tmp_path / "qwen3_asr_e2e"
    with RuntimeSession.open(
        device_index=0,
        artifact_dir=artifact_dir,
        model_dir=resolved_model_dir,
    ) as rt:
        rt.register_inputs(
            {
                tensors.input_features: input_features,
                tensors.feature_lens: feature_lens,
            }
        )
        run_qwen3_asr_audio_tower(rt, tensors)

        rt.register_inputs(
            {
                text_tensors.prefill.input_ids: prepared.input_ids,
                text_tensors.prefill.attention_mask: prepared.attention_mask,
                text_tensors.token_select.eos_token_ids: np.array(
                    QWEN3_ASR_DEFAULT_EOS_TOKEN_IDS,
                    dtype=np.int64,
                ),
            }
        )
        generated_tokens_tensor = run_qwen3_asr_greedy_decode_loop(
            rt,
            text_tensors,
            max_new_tokens=max_new_tokens,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )
        generated_array = rt.read_request_state(generated_tokens_tensor)
        generated_ids = tuple(int(t) for t in generated_array.flatten())

        failed = [r for r in rt.compare_results if not r.passed]
        all_compares = rt.compare_results

    decoded_text = _processor.batch_decode(
        torch.tensor([list(generated_ids)], dtype=torch.long),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(f"\nTranscription: {decoded_text!r}")
    print(f"Generated tokens: {generated_ids}")
    print(f"Total dispatches: {len(rt.dispatch_records)}")
    print(f"Total compares: {len(all_compares)}")
    print(f"Failed compares: {len(failed)}")

    if failed:
        print(f"\n{'=' * 60}")
        print("FAILED SHADER COMPARISONS:")
        print(f"{'=' * 60}")
        for r in failed:
            print(f"  Tensor: {r.artifact_key}")
            if r.drilldown_classification:
                print(f"    Classification: {r.drilldown_classification}")
            if r.drilldown_artifact_path:
                print(f"    Drilldown: {r.drilldown_artifact_path}")
        print(f"\nArtifact dir: {artifact_dir}")

    assert not failed, (
        f"{len(failed)} shader comparison(s) failed.\n"
        f"First failure: {failed[0].artifact_key}\n"
        f"Check drilldown artifacts in {artifact_dir}"
    )


def _expect_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _optional_mapping(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    return _expect_mapping(value, "optional mapping")


def _expect_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _expect_float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    return float(value)


def _int_tuple(value: object, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"expected integer sequence, got {type(value).__name__}")
    return tuple(_expect_int(item, "sequence item") for item in value)
