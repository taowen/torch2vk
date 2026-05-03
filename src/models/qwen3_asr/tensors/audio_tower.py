"""Logical tensors for Qwen3-ASR audio tower frames."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.runtime.logical import (
    ComparePolicy,
    LogicalTensor,
    MemoryClass,
    PyTorchProbe,
    TensorRole,
    TensorLifetime,
    TensorSpec,
    WeightSource,
)

QWEN3_ASR_CHECKPOINT = "model.safetensors"
CONV2D1_BIAS_KEY = "thinker.audio_tower.conv2d1.bias"
CONV2D1_WEIGHT_KEY = "thinker.audio_tower.conv2d1.weight"
QWEN3_ASR_WEIGHT_DTYPE = "bfloat16"
CONV2D1_BIAS_SHAPE = (480,)
CONV2D1_WEIGHT_SHAPE = (480, 1, 3, 3)


@dataclass(frozen=True, slots=True)
class Qwen3AsrAudioTowerWeights:
    conv2d1_weight: LogicalTensor
    conv2d1_bias: LogicalTensor

    def all(self) -> tuple[LogicalTensor, ...]:
        return (self.conv2d1_weight, self.conv2d1_bias)


@dataclass(frozen=True, slots=True)
class Qwen3AsrAudioTowerConv2d1Tensors:
    input_features: LogicalTensor
    weights: Qwen3AsrAudioTowerWeights
    conv2d1_gelu: LogicalTensor

    def dependencies(self) -> tuple[LogicalTensor, ...]:
        return self.weights.all()

    def all(self) -> tuple[LogicalTensor, ...]:
        return (self.input_features, *self.weights.all(), self.conv2d1_gelu)


def declare_qwen3_asr_audio_tower_conv2d1_tensors(
    *,
    input_shape: tuple[int, int, int, int],
    checkpoint: str = QWEN3_ASR_CHECKPOINT,
) -> Qwen3AsrAudioTowerConv2d1Tensors:
    if len(input_shape) != 4:
        raise ValueError(f"conv2d1 input_shape must be rank 4 NCHW, got {input_shape}")
    batch, input_channels, input_height, input_width = input_shape
    if min(input_shape) <= 0:
        raise ValueError(f"conv2d1 input_shape must be positive, got {input_shape}")

    weights = Qwen3AsrAudioTowerWeights(
        conv2d1_weight=_declare_weight(
            logical_name="qwen3_asr.audio_tower.conv2d1.weight",
            checkpoint=checkpoint,
            key=CONV2D1_WEIGHT_KEY,
            shape=CONV2D1_WEIGHT_SHAPE,
        ),
        conv2d1_bias=_declare_weight(
            logical_name="qwen3_asr.audio_tower.conv2d1.bias",
            checkpoint=checkpoint,
            key=CONV2D1_BIAS_KEY,
            shape=CONV2D1_BIAS_SHAPE,
        ),
    )

    out_channels, weight_input_channels, kernel_height, kernel_width = (
        int(dim) for dim in weights.conv2d1_weight.spec.shape
    )
    if input_channels != weight_input_channels:
        raise ValueError(
            f"conv2d1 input channel count {input_channels} does not match checkpoint "
            f"weight channels {weight_input_channels}"
        )
    if tuple(weights.conv2d1_bias.spec.shape) != (out_channels,):
        raise ValueError(
            f"conv2d1 bias shape {weights.conv2d1_bias.spec.shape} does not match out_channels {out_channels}"
        )

    output_height = _conv2d_output_dim(input_height, kernel_height, stride=2, padding=1)
    output_width = _conv2d_output_dim(input_width, kernel_width, stride=2, padding=1)
    return Qwen3AsrAudioTowerConv2d1Tensors(
        input_features=LogicalTensor(
            name="qwen3_asr.audio_tower.conv2d1.input",
            spec=TensorSpec(dtype="float32", shape=(batch, input_channels, input_height, input_width)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
        ),
        weights=weights,
        conv2d1_gelu=LogicalTensor(
            name="qwen3_asr.audio_tower.conv2d1.gelu",
            spec=TensorSpec(dtype="float32", shape=(batch, out_channels, output_height, output_width)),
            role=TensorRole.OUTPUT,
            memory=MemoryClass.HOST_OUTPUT,
            lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
            pytorch_probe=PyTorchProbe(kind="module_output", target=""),
        ),
    )


def _declare_weight(
    *,
    logical_name: str,
    checkpoint: str,
    key: str,
    shape: tuple[int, ...],
) -> LogicalTensor:
    return LogicalTensor(
        name=logical_name,
        spec=TensorSpec(dtype=QWEN3_ASR_WEIGHT_DTYPE, shape=shape),
        role=TensorRole.WEIGHT,
        memory=MemoryClass.MODEL_WEIGHT,
        lifetime=TensorLifetime.MODEL,
        source=WeightSource(
            checkpoint=checkpoint,
            key=key,
            dtype=QWEN3_ASR_WEIGHT_DTYPE,
            shape=shape,
        ),
    )


def _conv2d_output_dim(input_dim: int, kernel: int, *, stride: int, padding: int) -> int:
    output = ((input_dim + 2 * padding - kernel) // stride) + 1
    if output <= 0:
        raise ValueError(
            f"conv2d output dimension must be positive, got {output} for "
            f"input={input_dim}, kernel={kernel}, stride={stride}, padding={padding}"
        )
    return output
