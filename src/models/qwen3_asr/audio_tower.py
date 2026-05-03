"""Qwen3-ASR audio tower frames in PyTorch forward order."""

from __future__ import annotations

from dataclasses import dataclass

from models.qwen3_asr.shaders.audio_tower_conv2d1_gelu_f32 import (
    QWEN3_ASR_AUDIO_TOWER_CONV2D1_GELU_F32,
)
from models.qwen3_asr.tensors.audio_tower import Qwen3AsrAudioTowerConv2d1Tensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession


@dataclass(frozen=True, slots=True)
class Qwen3AsrAudioTowerConv2d1Output:
    output: LogicalTensor


def run_qwen3_asr_audio_tower_conv2d1_frame(
    rt: RuntimeSession,
    tensors: Qwen3AsrAudioTowerConv2d1Tensors,
    *,
    pytorch_model: object,
    pytorch_input: object,
) -> Qwen3AsrAudioTowerConv2d1Output:
    with rt.frame(
        "qwen3_asr.audio_tower.conv2d1",
        dependencies=tensors.dependencies(),
        pytorch_model=pytorch_model,
        pytorch_args=(pytorch_input,),
    ):
        QWEN3_ASR_AUDIO_TOWER_CONV2D1_GELU_F32(
            rt,
            x=tensors.input_features,
            weight=tensors.weights.conv2d1_weight,
            bias=tensors.weights.conv2d1_bias,
            output=tensors.conv2d1_gelu,
        )
    return Qwen3AsrAudioTowerConv2d1Output(output=tensors.conv2d1_gelu)
