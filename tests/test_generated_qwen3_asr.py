"""Generated Qwen3-ASR Vulkan adapter coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models.generated_qwen3_asr.audio_tower import run_generated_qwen3_asr_audio_tower
from models.generated_qwen3_asr.export import load_qwen3_asr_export_config
from models.generated_qwen3_asr.tensors.audio_tower import (
    declare_generated_qwen3_asr_audio_tower_tensors,
)
from models.hf_cache import resolve_cached_model
from models.qwen3_asr.execution import prepare_qwen3_asr_inputs
from models.qwen3_asr.pytorch.example import REPO_ID
from torch2vk.runtime.session import RuntimeSession


_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_ASKNOT_WAV = _FIXTURE_DIR / "qwen3_asr_asknot.wav"


def test_generated_qwen3_asr_audio_tower_runs_first_ops_and_matches_pytorch(
    tmp_path: Path,
) -> None:
    model_dir = resolve_cached_model(REPO_ID)
    config = load_qwen3_asr_export_config(model_dir=model_dir)
    _, prepared = prepare_qwen3_asr_inputs(
        model_dir=model_dir,
        wav=_ASKNOT_WAV,
        language="English",
    )
    audio_feature_len = prepared.audio_feature_length
    input_features = np.ascontiguousarray(
        prepared.input_features[0, :, :audio_feature_len],
        dtype=np.float32,
    )
    feature_lens = np.array([audio_feature_len], dtype=np.int64)

    tensors = declare_generated_qwen3_asr_audio_tower_tensors(
        input_features_shape=input_features.shape,
        hidden_size=config.audio_hidden_size,
        output_size=config.audio_output_size,
        downsample_hidden_size=config.audio_downsample_hidden_size,
        encoder_layers=config.audio_encoder_layers,
        encoder_ffn_dim=config.audio_encoder_ffn_dim,
    )

    with RuntimeSession.open(
        artifact_dir=tmp_path,
        model_dir=model_dir,
    ) as rt:
        rt.register_inputs(
            {
                tensors.input_features: input_features,
                tensors.feature_lens: feature_lens,
            }
        )

        assert (
            run_generated_qwen3_asr_audio_tower(
                rt, tensors, pytorch_compare=True, max_ops=5
            )
            is tensors.conv_out
        )

        assert [record.shader for record in rt.dispatch_records[:5]] == [
            "qwen3_asr_pad_feature_f32",
            "qwen3_asr_audio_tower_conv2d_gelu_f32",
            "qwen3_asr_audio_tower_conv2d_gelu_f32",
            "qwen3_asr_audio_tower_conv2d_gelu_f32",
            "qwen3_asr_audio_tower_conv_out_f32",
        ]
        assert rt.compare_results
        assert all(result.passed for result in rt.compare_results)
