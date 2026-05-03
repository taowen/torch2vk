"""Qwen3-ASR eager runtime entry points."""

from __future__ import annotations

from models.qwen3_asr.shaders.conv2d_gelu_f32 import QWEN3_ASR_CONV2D_GELU_F32
from models.qwen3_asr.shaders.conv_out_add_position_f32 import QWEN3_ASR_CONV_OUT_ADD_POSITION_F32
from models.qwen3_asr.tensors.audio_tower import (
    Qwen3AsrAudioTowerTensors,
)
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession


def run_qwen3_asr_audio_tower(
    rt: RuntimeSession,
    tensors: Qwen3AsrAudioTowerTensors,
    *,
    pytorch_model: object,
) -> LogicalTensor:
    with rt.frame(
        "qwen3_asr.audio_tower",
        scope={
            "chunks": tensors.padded_feature.spec.shape[0],
            "frames": tensors.padded_feature.spec.shape[3],
        },
        pytorch_model=pytorch_model,
    ):
        QWEN3_ASR_CONV2D_GELU_F32(
            rt,
            x=tensors.padded_feature,
            weight=tensors.weights.conv2d1_weight,
            bias=tensors.weights.conv2d1_bias,
            output=tensors.conv2d1_gelu,
        )
        QWEN3_ASR_CONV2D_GELU_F32(
            rt,
            x=tensors.conv2d1_gelu,
            weight=tensors.weights.conv2d2_weight,
            bias=tensors.weights.conv2d2_bias,
            output=tensors.conv2d2_gelu,
        )
        QWEN3_ASR_CONV2D_GELU_F32(
            rt,
            x=tensors.conv2d2_gelu,
            weight=tensors.weights.conv2d3_weight,
            bias=tensors.weights.conv2d3_bias,
            output=tensors.conv2d3_gelu,
        )
        QWEN3_ASR_CONV_OUT_ADD_POSITION_F32(
            rt,
            x=tensors.conv2d3_gelu,
            weight=tensors.weights.conv_out_weight,
            output=tensors.conv_out_add_position,
        )
    return tensors.conv_out_add_position
