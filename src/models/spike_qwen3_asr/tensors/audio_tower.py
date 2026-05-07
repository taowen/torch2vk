"""Generated tensor declarations."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.runtime.logical import (
    ComparePolicy,
    LogicalTensor,
    MemoryClass,
    PyTorchProbe,
    TensorLifetime,
    TensorRole,
)
from torch2vk.vulkan.types import TensorSpec


@dataclass(frozen=True, slots=True)
class Conv2d1Tensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    input: LogicalTensor
    conv2d: LogicalTensor


CONV2D1_WEIGHT_MAP: dict[str, str] = {
    'p_weight': 'thinker.audio_tower.conv2d1.weight',
    'p_bias': 'thinker.audio_tower.conv2d1.bias',
}

CONV2D1_OUTPUT: str = 'conv2d'


def create_conv2d1(prefix: str) -> Conv2d1Tensors:
    return Conv2d1Tensors(
        p_weight=LogicalTensor(
            name=f"{prefix}.p_weight",
            spec=TensorSpec(dtype='float32', shape=(480, 1, 3, 3)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_bias=LogicalTensor(
            name=f"{prefix}.p_bias",
            spec=TensorSpec(dtype='float32', shape=(480,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        input=LogicalTensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='float32', shape=(11, 1, 128, 100)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        conv2d=LogicalTensor(
            name=f"{prefix}.conv2d",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 64, 50)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )


@dataclass(frozen=True, slots=True)
class Conv2d2Tensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    input: LogicalTensor
    conv2d: LogicalTensor


CONV2D2_WEIGHT_MAP: dict[str, str] = {
    'p_weight': 'thinker.audio_tower.conv2d2.weight',
    'p_bias': 'thinker.audio_tower.conv2d2.bias',
}

CONV2D2_OUTPUT: str = 'conv2d'


def create_conv2d2(prefix: str) -> Conv2d2Tensors:
    return Conv2d2Tensors(
        p_weight=LogicalTensor(
            name=f"{prefix}.p_weight",
            spec=TensorSpec(dtype='float32', shape=(480, 480, 3, 3)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_bias=LogicalTensor(
            name=f"{prefix}.p_bias",
            spec=TensorSpec(dtype='float32', shape=(480,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        input=LogicalTensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 64, 50)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        conv2d=LogicalTensor(
            name=f"{prefix}.conv2d",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 32, 25)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )


@dataclass(frozen=True, slots=True)
class Conv2d3Tensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    input: LogicalTensor
    conv2d: LogicalTensor


CONV2D3_WEIGHT_MAP: dict[str, str] = {
    'p_weight': 'thinker.audio_tower.conv2d3.weight',
    'p_bias': 'thinker.audio_tower.conv2d3.bias',
}

CONV2D3_OUTPUT: str = 'conv2d'


def create_conv2d3(prefix: str) -> Conv2d3Tensors:
    return Conv2d3Tensors(
        p_weight=LogicalTensor(
            name=f"{prefix}.p_weight",
            spec=TensorSpec(dtype='float32', shape=(480, 480, 3, 3)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_bias=LogicalTensor(
            name=f"{prefix}.p_bias",
            spec=TensorSpec(dtype='float32', shape=(480,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        input=LogicalTensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 32, 25)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        conv2d=LogicalTensor(
            name=f"{prefix}.conv2d",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 16, 13)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )


@dataclass(frozen=True, slots=True)
class ConvOutTensors:
    p_weight: LogicalTensor
    input: LogicalTensor
    linear: LogicalTensor


CONV_OUT_WEIGHT_MAP: dict[str, str] = {
    'p_weight': 'thinker.audio_tower.conv_out.weight',
}

CONV_OUT_OUTPUT: str = 'linear'


def create_conv_out(prefix: str) -> ConvOutTensors:
    return ConvOutTensors(
        p_weight=LogicalTensor(
            name=f"{prefix}.p_weight",
            spec=TensorSpec(dtype='float32', shape=(896, 7680)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        input=LogicalTensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='float32', shape=(143, 7680)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        linear=LogicalTensor(
            name=f"{prefix}.linear",
            spec=TensorSpec(dtype='float32', shape=(143, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )


@dataclass(frozen=True, slots=True)
class LnPostTensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    input: LogicalTensor
    layer_norm: LogicalTensor


LN_POST_WEIGHT_MAP: dict[str, str] = {
    'p_weight': 'thinker.audio_tower.ln_post.weight',
    'p_bias': 'thinker.audio_tower.ln_post.bias',
}

LN_POST_OUTPUT: str = 'layer_norm'


def create_ln_post(prefix: str) -> LnPostTensors:
    return LnPostTensors(
        p_weight=LogicalTensor(
            name=f"{prefix}.p_weight",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_bias=LogicalTensor(
            name=f"{prefix}.p_bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        input=LogicalTensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        layer_norm=LogicalTensor(
            name=f"{prefix}.layer_norm",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )


@dataclass(frozen=True, slots=True)
class Proj1Tensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    input: LogicalTensor
    linear: LogicalTensor


PROJ1_WEIGHT_MAP: dict[str, str] = {
    'p_weight': 'thinker.audio_tower.proj1.weight',
    'p_bias': 'thinker.audio_tower.proj1.bias',
}

PROJ1_OUTPUT: str = 'linear'


def create_proj1(prefix: str) -> Proj1Tensors:
    return Proj1Tensors(
        p_weight=LogicalTensor(
            name=f"{prefix}.p_weight",
            spec=TensorSpec(dtype='float32', shape=(896, 896)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_bias=LogicalTensor(
            name=f"{prefix}.p_bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        input=LogicalTensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        linear=LogicalTensor(
            name=f"{prefix}.linear",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )


@dataclass(frozen=True, slots=True)
class Proj2Tensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    input: LogicalTensor
    linear: LogicalTensor


PROJ2_WEIGHT_MAP: dict[str, str] = {
    'p_weight': 'thinker.audio_tower.proj2.weight',
    'p_bias': 'thinker.audio_tower.proj2.bias',
}

PROJ2_OUTPUT: str = 'linear'


def create_proj2(prefix: str) -> Proj2Tensors:
    return Proj2Tensors(
        p_weight=LogicalTensor(
            name=f"{prefix}.p_weight",
            spec=TensorSpec(dtype='float32', shape=(1024, 896)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        p_bias=LogicalTensor(
            name=f"{prefix}.p_bias",
            spec=TensorSpec(dtype='float32', shape=(1024,)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        input=LogicalTensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        linear=LogicalTensor(
            name=f"{prefix}.linear",
            spec=TensorSpec(dtype='float32', shape=(133, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )

