"""Logical tensor declarations owned by model adapters and materialized by runtime."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Literal

from torch2vk.vulkan.allocation import BufferSlice
from torch2vk.vulkan.types import CONTIGUOUS_LAYOUT, TensorLayout, TensorSpec, dtype_nbytes, validate_tensor_layout


class TensorRole(StrEnum):
    MISSING_VALUE = "missing_value"
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
    MISSING_VALUE = "missing_value"
    MODEL_WEIGHT = "model_weight"
    REQUEST_STATE = "request_state"
    FRAME_WORKSPACE = "frame_workspace"
    OP_SCRATCH = "op_scratch"
    HOST_INPUT = "host_input"
    HOST_OUTPUT = "host_output"


class TensorLifetime(StrEnum):
    MISSING_VALUE = "missing_value"
    MODEL = "model"
    REQUEST = "request"
    FRAME = "frame"
    OP = "op"
    EXTERNAL = "external"


def default_memory_lifetime_for_role(
    role: TensorRole,
) -> tuple[MemoryClass, TensorLifetime] | None:
    if role is TensorRole.WEIGHT:
        return MemoryClass.MODEL_WEIGHT, TensorLifetime.MODEL
    if role is TensorRole.INPUT:
        return MemoryClass.HOST_INPUT, TensorLifetime.FRAME
    if role is TensorRole.ACTIVATION:
        return MemoryClass.FRAME_WORKSPACE, TensorLifetime.FRAME
    if role is TensorRole.SCRATCH:
        return MemoryClass.OP_SCRATCH, TensorLifetime.OP
    if role is TensorRole.OUTPUT:
        return MemoryClass.HOST_OUTPUT, TensorLifetime.FRAME
    if role is TensorRole.STATE:
        return MemoryClass.REQUEST_STATE, TensorLifetime.REQUEST
    return None


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


@dataclass(frozen=True, slots=True)
class DispatchWriter:
    frame: str
    shader: str
    dispatch_index: int


@dataclass(slots=True, eq=False)
class LogicalTensor:
    name: str
    spec: TensorSpec
    role: TensorRole
    memory: MemoryClass
    lifetime: TensorLifetime
    layout: TensorLayout = CONTIGUOUS_LAYOUT
    semantic: TensorSemantic | None = None
    compare: ComparePolicy | None = None
    pytorch_probe: PyTorchProbe | None = None
    checkpoint: str | None = None
    _runtime_writable: bool = field(default=False, init=False, repr=False)
    _buffer: BufferSlice | None = field(default=None, init=False, repr=False)
    _descriptor_nbytes: int | None = field(default=None, init=False, repr=False)
    _version: int = field(default=0, init=False, repr=False)
    _writer: DispatchWriter | None = field(default=None, init=False, repr=False)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "spec" and hasattr(self, "_runtime_writable"):
            self._require_runtime_writable("spec")
        object.__setattr__(self, name, value)


    @property
    def runtime_writable(self) -> bool:
        return self._runtime_writable

    @contextmanager
    def runtime_write_scope(self) -> Iterator[None]:
        previous = self._runtime_writable
        self._runtime_writable = True
        try:
            yield
        finally:
            self._runtime_writable = previous

    @property
    def buffer(self) -> BufferSlice | None:
        return self._buffer

    @buffer.setter
    def buffer(self, value: BufferSlice | None) -> None:
        self._require_runtime_writable("buffer")
        self._buffer = value

    @property
    def descriptor_nbytes(self) -> int | None:
        return self._descriptor_nbytes

    @descriptor_nbytes.setter
    def descriptor_nbytes(self, value: int | None) -> None:
        self._require_runtime_writable("descriptor_nbytes")
        self._descriptor_nbytes = value

    @property
    def version(self) -> int:
        return self._version

    @version.setter
    def version(self, value: int) -> None:
        self._require_runtime_writable("version")
        self._version = value

    @property
    def writer(self) -> DispatchWriter | None:
        return self._writer

    @writer.setter
    def writer(self, value: DispatchWriter | None) -> None:
        self._require_runtime_writable("writer")
        self._writer = value

    def validate_declaration(self) -> None:
        self.fill_missing_declaration_defaults()
        if not self.name:
            raise ValueError("LogicalTensor name must be non-empty")
        if any(part == "" for part in self.name.split(".")):
            raise ValueError(f"LogicalTensor name has an empty component: {self.name!r}")
        if (
            self.role is TensorRole.MISSING_VALUE
            or self.memory is MemoryClass.MISSING_VALUE
            or self.lifetime is TensorLifetime.MISSING_VALUE
            or self.spec.dtype == "missing_value"
            or "MISSING_VALUE" in self.spec.shape
        ):
            raise ValueError(f"{self.name} has unresolved missing LogicalTensor metadata")
        dtype_nbytes(self.spec.dtype)
        if not self.spec.shape:
            raise ValueError(f"{self.name} shape must have fixed rank")
        validate_tensor_layout(self.layout, self.spec.shape)
        _validate_role_memory_lifetime(self)

    def fill_missing_declaration_defaults(self) -> None:
        defaults = default_memory_lifetime_for_role(self.role)
        if defaults is None:
            return
        memory, lifetime = defaults
        if self.memory is MemoryClass.MISSING_VALUE:
            self.memory = memory
        if self.lifetime is TensorLifetime.MISSING_VALUE:
            self.lifetime = lifetime

    @property
    def concrete_shape(self) -> tuple[int, ...]:
        dims: list[int] = []
        for dim in self.spec.shape:
            if not isinstance(dim, int):
                raise ValueError(f"{self.name} has unresolved symbolic shape {self.spec.shape}")
            dims.append(dim)
        return tuple(dims)

    def _require_runtime_writable(self, field_name: str) -> None:
        if not self._runtime_writable:
            raise RuntimeError(
                f"{self.name}.{field_name} is runtime state and cannot be written during "
                "LogicalTensor declaration"
            )


def _validate_role_memory_lifetime(tensor: LogicalTensor) -> None:
    role = tensor.role
    memory = tensor.memory
    lifetime = tensor.lifetime
    if (
        role is TensorRole.MISSING_VALUE
        or memory is MemoryClass.MISSING_VALUE
        or lifetime is TensorLifetime.MISSING_VALUE
    ):
        raise ValueError(f"{tensor.name} has unresolved missing LogicalTensor metadata")
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
    if role is TensorRole.WEIGHT and (
        memory is not MemoryClass.MODEL_WEIGHT or lifetime is not TensorLifetime.MODEL
    ):
        raise ValueError(f"{tensor.name} weight must use MODEL_WEIGHT/MODEL")
    if role is TensorRole.OUTPUT:
        if memory not in {MemoryClass.FRAME_WORKSPACE, MemoryClass.HOST_OUTPUT, MemoryClass.REQUEST_STATE}:
            raise ValueError(f"{tensor.name} output cannot use memory={memory}")
        if lifetime not in {TensorLifetime.FRAME, TensorLifetime.REQUEST}:
            raise ValueError(f"{tensor.name} output cannot use lifetime={lifetime}")
    if role is TensorRole.INPUT:
        if memory not in {MemoryClass.HOST_INPUT, MemoryClass.REQUEST_STATE}:
            raise ValueError(f"{tensor.name} input cannot use memory={memory}")
        if lifetime not in {TensorLifetime.FRAME, TensorLifetime.REQUEST}:
            raise ValueError(f"{tensor.name} input cannot use lifetime={lifetime}")
