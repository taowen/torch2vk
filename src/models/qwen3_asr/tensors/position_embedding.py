"""Logical tensors for the Qwen3-ASR audio position embedding frame."""

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
)


@dataclass(frozen=True, slots=True)
class Qwen3AsrPositionEmbeddingTensors:
    output: LogicalTensor

    def all(self) -> tuple[LogicalTensor, ...]:
        return (self.output,)


def declare_qwen3_asr_position_embedding_tensors(
    *,
    seqlen: int,
    channels: int,
) -> Qwen3AsrPositionEmbeddingTensors:
    if seqlen <= 0:
        raise ValueError(f"seqlen must be positive, got {seqlen}")
    if channels < 4 or channels % 2 != 0:
        raise ValueError(f"channels must be an even integer >= 4, got {channels}")
    return Qwen3AsrPositionEmbeddingTensors(
        output=LogicalTensor(
            name="qwen3_asr.audio_tower.position_embedding.output",
            spec=TensorSpec(dtype="float32", shape=(seqlen, channels)),
            role=TensorRole.OUTPUT,
            memory=MemoryClass.HOST_OUTPUT,
            lifetime=TensorLifetime.FRAME,
            compare=ComparePolicy(kind="tensor", rtol=1e-5, atol=1e-6),
            pytorch_probe=PyTorchProbe(
                kind="module_output",
                target="",
            ),
        )
    )
