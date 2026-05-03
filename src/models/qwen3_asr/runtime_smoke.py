"""Minimal Qwen3-ASR runtime frame for validating torch2vk.runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models.qwen3_asr.shaders.weight_copy_bf16 import WEIGHT_COPY_BF16
from torch2vk.checkpoints.safetensors import open_safetensors_mmap
from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    WeightSource,
)
from torch2vk.runtime.session import RuntimeSession
from torch2vk.vulkan.types import TensorSpec

QWEN3_ASR_SMOKE_WEIGHT_KEY = "thinker.audio_tower.conv2d1.bias"
QWEN3_ASR_SMOKE_CHECKPOINT = "model.safetensors"


@dataclass(frozen=True, slots=True)
class Qwen3AsrRuntimeSmokeTensors:
    weight: LogicalTensor
    output: LogicalTensor

    def all(self) -> tuple[LogicalTensor, ...]:
        return (self.weight, self.output)


def declare_qwen3_asr_runtime_smoke_tensors(model_dir: str | Path) -> Qwen3AsrRuntimeSmokeTensors:
    resolved_model_dir = Path(model_dir).expanduser().resolve()
    checkpoint = resolved_model_dir / QWEN3_ASR_SMOKE_CHECKPOINT
    with open_safetensors_mmap(checkpoint) as storage:
        entry = storage.entry(QWEN3_ASR_SMOKE_WEIGHT_KEY)
        dtype = entry.spec.dtype
        shape = entry.shape

    weight = LogicalTensor(
        name="qwen3_asr.smoke.audio_tower.conv2d1.bias",
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.WEIGHT,
        memory=MemoryClass.MODEL_WEIGHT,
        lifetime=TensorLifetime.MODEL,
        source=WeightSource(
            checkpoint=QWEN3_ASR_SMOKE_CHECKPOINT,
            key=QWEN3_ASR_SMOKE_WEIGHT_KEY,
            dtype=dtype,
            shape=shape,
        ),
    )
    output = LogicalTensor(
        name="qwen3_asr.smoke.audio_tower.conv2d1.bias.copy",
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.OUTPUT,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )
    return Qwen3AsrRuntimeSmokeTensors(weight=weight, output=output)


def run_qwen3_asr_runtime_smoke_frame(
    rt: RuntimeSession,
    tensors: Qwen3AsrRuntimeSmokeTensors,
) -> np.ndarray:
    with rt.frame("qwen3_asr.runtime_smoke", scope={"weight": "conv2d1.bias"}):
        WEIGHT_COPY_BF16(rt, weight=tensors.weight, output=tensors.output)
    return rt.readback(tensors.output)
