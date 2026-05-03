"""Logical tensors for Qwen3-ASR audio tower frames."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.runtime.logical import (
    ComparePolicy,
    LogicalTensor,
    MemoryClass,
    PyTorchProbe,
    TensorLifetime,
    TensorRole,
    TensorSpec,
)


@dataclass(frozen=True, slots=True)
class Qwen3AsrAudioTowerWeights:
    conv2d1_weight: LogicalTensor
    conv2d1_bias: LogicalTensor
    conv2d2_weight: LogicalTensor
    conv2d2_bias: LogicalTensor
    conv2d3_weight: LogicalTensor
    conv2d3_bias: LogicalTensor
    conv_out_weight: LogicalTensor


@dataclass(frozen=True, slots=True)
class Qwen3AsrAudioTowerTensors:
    input_features: LogicalTensor
    feature_lens: LogicalTensor
    padded_feature: LogicalTensor
    weights: Qwen3AsrAudioTowerWeights
    conv2d1_gelu: LogicalTensor
    conv2d2_gelu: LogicalTensor
    conv2d3_gelu: LogicalTensor
    conv_out_add_position: LogicalTensor


def declare_qwen3_asr_audio_tower_tensors(
    *,
    input_shape: tuple[int, int, int, int],
    hidden_size: int = 896,
    downsample_hidden_size: int = 480,
) -> Qwen3AsrAudioTowerTensors:
    batch, _, input_height, input_width = input_shape
    conv1_height, conv1_width = (input_height + 1) // 2, (input_width + 1) // 2
    conv2_height, conv2_width = (conv1_height + 1) // 2, (conv1_width + 1) // 2
    conv3_height, conv3_width = (conv2_height + 1) // 2, (conv2_width + 1) // 2
    conv_out_input_features = downsample_hidden_size * conv3_height
    weight_common = {
        "role": TensorRole.WEIGHT,
        "memory": MemoryClass.MODEL_WEIGHT,
        "lifetime": TensorLifetime.MODEL,
    }
    output_common = {
        "role": TensorRole.OUTPUT,
        "memory": MemoryClass.HOST_OUTPUT,
        "lifetime": TensorLifetime.FRAME,
    }

    weights = Qwen3AsrAudioTowerWeights(
        conv2d1_weight=LogicalTensor(
            name="thinker.audio_tower.conv2d1.weight",
            spec=TensorSpec(dtype="bfloat16", shape=(downsample_hidden_size, 1, 3, 3)),
            **weight_common,
        ),
        conv2d1_bias=LogicalTensor(
            name="thinker.audio_tower.conv2d1.bias",
            spec=TensorSpec(dtype="bfloat16", shape=(downsample_hidden_size,)),
            **weight_common,
        ),
        conv2d2_weight=LogicalTensor(
            name="thinker.audio_tower.conv2d2.weight",
            spec=TensorSpec(dtype="bfloat16", shape=(downsample_hidden_size, downsample_hidden_size, 3, 3)),
            **weight_common,
        ),
        conv2d2_bias=LogicalTensor(
            name="thinker.audio_tower.conv2d2.bias",
            spec=TensorSpec(dtype="bfloat16", shape=(downsample_hidden_size,)),
            **weight_common,
        ),
        conv2d3_weight=LogicalTensor(
            name="thinker.audio_tower.conv2d3.weight",
            spec=TensorSpec(dtype="bfloat16", shape=(downsample_hidden_size, downsample_hidden_size, 3, 3)),
            **weight_common,
        ),
        conv2d3_bias=LogicalTensor(
            name="thinker.audio_tower.conv2d3.bias",
            spec=TensorSpec(dtype="bfloat16", shape=(downsample_hidden_size,)),
            **weight_common,
        ),
        conv_out_weight=LogicalTensor(
            name="thinker.audio_tower.conv_out.weight",
            spec=TensorSpec(dtype="bfloat16", shape=(hidden_size, conv_out_input_features)),
            **weight_common,
        ),
    )

    return Qwen3AsrAudioTowerTensors(
        input_features=LogicalTensor(
            name="qwen3_asr.audio_tower.input_features",
            spec=TensorSpec(dtype="float32", shape=(input_height, input_width)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
        ),
        feature_lens=LogicalTensor(
            name="qwen3_asr.audio_tower.feature_lens",
            spec=TensorSpec(dtype="int64", shape=(batch,)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
        ),
        padded_feature=LogicalTensor(
            name="qwen3_asr.audio_tower.padded_feature",
            spec=TensorSpec(dtype="float32", shape=(batch, 1, input_height, input_width)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
        ),
        weights=weights,
        conv2d1_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.conv2d1.gelu",
            spec=TensorSpec(dtype="float32", shape=(batch, downsample_hidden_size, conv1_height, conv1_width)),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
            pytorch_probe=PyTorchProbe(kind="module_input", target="conv2d2", index=0),
            **output_common,
        ),
        conv2d2_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.conv2d2.gelu",
            spec=TensorSpec(dtype="float32", shape=(batch, downsample_hidden_size, conv2_height, conv2_width)),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
            pytorch_probe=PyTorchProbe(kind="module_input", target="conv2d3", index=0),
            **output_common,
        ),
        conv2d3_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.conv2d3.gelu",
            spec=TensorSpec(dtype="float32", shape=(batch, downsample_hidden_size, conv3_height, conv3_width)),
            **output_common,
        ),
        conv_out_add_position=LogicalTensor(
            name="qwen3_asr.audio_tower.conv_out.add_position",
            spec=TensorSpec(dtype="float32", shape=(batch * conv3_width, hidden_size)),
            compare=ComparePolicy(kind="tensor", rtol=2e-3, atol=2e-2),
            pytorch_probe=PyTorchProbe(kind="module_input", target="ln_post", index=0),
            **output_common,
        ),
    )
