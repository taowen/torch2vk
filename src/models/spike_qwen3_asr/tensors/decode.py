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
class DecodeEmbedTensors:
    p_weight: LogicalTensor
    input: LogicalTensor
    embedding: LogicalTensor


DECODE_EMBED_OUTPUT: str = 'embedding'


def create_decode_embed(prefix: str) -> DecodeEmbedTensors:
    return DecodeEmbedTensors(
        p_weight=LogicalTensor(
            name="thinker.model.embed_tokens.weight",
            spec=TensorSpec(dtype='float32', shape=(151936, 1024)),
            role=TensorRole.WEIGHT, memory=MemoryClass.MODEL_WEIGHT, lifetime=TensorLifetime.MODEL,
        ),
        input=LogicalTensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='int32', shape=(1, 1)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        embedding=LogicalTensor(
            name=f"{prefix}.embedding",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )


@dataclass(frozen=True, slots=True)
class DecodeNormTensors:
    p_weight: LogicalTensor
    hidden_states: LogicalTensor
    to: LogicalTensor
    pow_1: LogicalTensor
    mean: LogicalTensor
    add: LogicalTensor
    rsqrt: LogicalTensor
    mul: LogicalTensor
    to_1: LogicalTensor
    mul_1: LogicalTensor


DECODE_NORM_OUTPUT: str = 'mul_1'


def create_decode_norm(prefix: str) -> DecodeNormTensors:
    return DecodeNormTensors(
        p_weight=LogicalTensor(
            name="thinker.model.norm.weight",
            spec=TensorSpec(dtype='float32', shape=(1024,)),
            role=TensorRole.WEIGHT, memory=MemoryClass.MODEL_WEIGHT, lifetime=TensorLifetime.MODEL,
        ),
        hidden_states=LogicalTensor(
            name=f"{prefix}.hidden_states",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        to=LogicalTensor(
            name=f"{prefix}.to",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        pow_1=LogicalTensor(
            name=f"{prefix}.pow_1",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mean=LogicalTensor(
            name=f"{prefix}.mean",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        add=LogicalTensor(
            name=f"{prefix}.add",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        rsqrt=LogicalTensor(
            name=f"{prefix}.rsqrt",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 1)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul=LogicalTensor(
            name=f"{prefix}.mul",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        to_1=LogicalTensor(
            name=f"{prefix}.to_1",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
        ),
        mul_1=LogicalTensor(
            name=f"{prefix}.mul_1",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )


@dataclass(frozen=True, slots=True)
class DecodeLmHeadTensors:
    p_weight: LogicalTensor
    input: LogicalTensor
    linear: LogicalTensor


DECODE_LM_HEAD_OUTPUT: str = 'linear'


def create_decode_lm_head(prefix: str) -> DecodeLmHeadTensors:
    return DecodeLmHeadTensors(
        p_weight=LogicalTensor(
            name="thinker.lm_head.weight",
            spec=TensorSpec(dtype='float32', shape=(151936, 1024)),
            role=TensorRole.WEIGHT, memory=MemoryClass.MODEL_WEIGHT, lifetime=TensorLifetime.MODEL,
        ),
        input=LogicalTensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
            role=TensorRole.INPUT, memory=MemoryClass.HOST_INPUT, lifetime=TensorLifetime.FRAME,
        ),
        linear=LogicalTensor(
            name=f"{prefix}.linear",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 151936)),
            role=TensorRole.ACTIVATION, memory=MemoryClass.FRAME_WORKSPACE, lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
        ),
    )

