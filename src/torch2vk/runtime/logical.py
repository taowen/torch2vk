"""Logical tensor declarations owned by model adapters and materialized by runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from torch2vk.vulkan.allocation import BufferSlice
from torch2vk.vulkan.types import CONTIGUOUS_LAYOUT, TensorLayout, TensorSpec, dtype_nbytes


class TensorRole(StrEnum):
    INPUT = "input"
    WEIGHT = "weight"
    ACTIVATION = "activation"
    SCRATCH = "scratch"
    OUTPUT = "output"
    STATE = "state"


class TensorSemantic(StrEnum):
    LOGITS = "logits"
    TOKEN = "token"
    KV_CACHE = "kv_cache"
    MASK = "mask"
    WAVEFORM = "waveform"


class MemoryClass(StrEnum):
    MODEL_WEIGHT = "model_weight"
    REQUEST_STATE = "request_state"
    FRAME_WORKSPACE = "frame_workspace"
    OP_SCRATCH = "op_scratch"
    HOST_INPUT = "host_input"
    HOST_OUTPUT = "host_output"


class TensorLifetime(StrEnum):
    MODEL = "model"
    REQUEST = "request"
    FRAME = "frame"
    OP = "op"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class WeightSource:
    checkpoint: str | Path
    key: str
    dtype: str
    shape: tuple[int, ...]
    layout: TensorLayout = CONTIGUOUS_LAYOUT


@dataclass(frozen=True, slots=True)
class InputFeed:
    name: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class ComparePolicy:
    kind: Literal["tensor", "token", "waveform"] = "tensor"
    rtol: float = 1e-4
    atol: float = 1e-4
    max_abs: float | None = None


@dataclass(frozen=True, slots=True)
class PyTorchProbe:
    kind: Literal["module_input", "module_output", "manual_hook", "derived"]
    target: str
    index: int = 0
    selector: str | None = None
    transform: str | None = None


@dataclass(frozen=True, slots=True)
class DispatchWriter:
    frame: str
    shader: str
    dispatch_index: int


@dataclass(slots=True)
class LogicalTensor:
    name: str
    spec: TensorSpec
    role: TensorRole
    memory: MemoryClass
    lifetime: TensorLifetime
    layout: TensorLayout = CONTIGUOUS_LAYOUT
    source: WeightSource | None = None
    feed: InputFeed | None = None
    semantic: TensorSemantic | None = None
    compare: ComparePolicy | None = None
    pytorch_probe: PyTorchProbe | None = None
    buffer: BufferSlice | None = None
    descriptor_nbytes: int | None = None
    version: int = 0
    writer: DispatchWriter | None = None

    def validate_declaration(self) -> None:
        if not self.name:
            raise ValueError("LogicalTensor name must be non-empty")
        if any(part == "" for part in self.name.split(".")):
            raise ValueError(f"LogicalTensor name has an empty component: {self.name!r}")
        dtype_nbytes(self.spec.dtype)
        if not self.spec.shape:
            raise ValueError(f"{self.name} shape must have fixed rank")
        if self.source is not None:
            if self.memory is not MemoryClass.MODEL_WEIGHT:
                raise ValueError(f"{self.name} has WeightSource but memory={self.memory}")
            if self.lifetime is not TensorLifetime.MODEL:
                raise ValueError(f"{self.name} has WeightSource but lifetime={self.lifetime}")
            if self.source.dtype != self.spec.dtype:
                raise ValueError(
                    f"{self.name} source dtype {self.source.dtype} does not match spec dtype {self.spec.dtype}"
                )
            if tuple(self.source.shape) != tuple(self.spec.shape):
                raise ValueError(
                    f"{self.name} source shape {self.source.shape} does not match spec shape {self.spec.shape}"
                )
        if self.role is TensorRole.WEIGHT and self.source is None:
            raise ValueError(f"{self.name} is a WEIGHT tensor but has no WeightSource")
        if self.feed is not None and not self.feed.name:
            raise ValueError(f"{self.name} feed name must be non-empty")
        _validate_role_memory_lifetime(self)

    @property
    def concrete_shape(self) -> tuple[int, ...]:
        dims: list[int] = []
        for dim in self.spec.shape:
            if not isinstance(dim, int):
                raise ValueError(f"{self.name} has unresolved symbolic shape {self.spec.shape}")
            dims.append(dim)
        return tuple(dims)


def _validate_role_memory_lifetime(tensor: LogicalTensor) -> None:
    role = tensor.role
    memory = tensor.memory
    lifetime = tensor.lifetime
    if role is TensorRole.ACTIVATION and (
        memory is not MemoryClass.FRAME_WORKSPACE or lifetime is not TensorLifetime.FRAME
    ):
        raise ValueError(f"{tensor.name} activation must use FRAME_WORKSPACE/FRAME")
    if role is TensorRole.SCRATCH and (
        memory is not MemoryClass.OP_SCRATCH or lifetime is not TensorLifetime.OP
    ):
        raise ValueError(f"{tensor.name} scratch must use OP_SCRATCH/OP")
    if role is TensorRole.STATE and (
        memory is not MemoryClass.REQUEST_STATE or lifetime is not TensorLifetime.REQUEST
    ):
        raise ValueError(f"{tensor.name} state must use REQUEST_STATE/REQUEST")
    if role is TensorRole.OUTPUT:
        if memory not in {MemoryClass.FRAME_WORKSPACE, MemoryClass.HOST_OUTPUT, MemoryClass.REQUEST_STATE}:
            raise ValueError(f"{tensor.name} output cannot use memory={memory}")
        if lifetime not in {TensorLifetime.FRAME, TensorLifetime.REQUEST}:
            raise ValueError(f"{tensor.name} output cannot use lifetime={lifetime}")
    if role is TensorRole.INPUT and tensor.feed is not None:
        if memory not in {MemoryClass.HOST_INPUT, MemoryClass.REQUEST_STATE}:
            raise ValueError(f"{tensor.name} input feed cannot use memory={memory}")
        if lifetime not in {TensorLifetime.FRAME, TensorLifetime.REQUEST}:
            raise ValueError(f"{tensor.name} input feed cannot use lifetime={lifetime}")
