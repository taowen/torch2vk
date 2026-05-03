"""Logical tensor identities used by model execution code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any


class TensorRole(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    WEIGHT = "weight"
    ACTIVATION = "activation"
    SCRATCH = "scratch"
    KV_CACHE = "kv_cache"
    RECURRENT_STATE = "recurrent_state"
    MASK = "mask"
    LOGITS = "logits"
    DEBUG = "debug"
    TOKEN = "token"


class MemoryPolicy(StrEnum):
    DEVICE_LOCAL = "device_local"
    HOST_VISIBLE_INPUT = "host_visible_input"
    HOST_VISIBLE_OUTPUT = "host_visible_output"
    PERSISTENT_STATE = "persistent_state"
    FRAME_WORKSPACE = "frame_workspace"
    STEP_TEMPORARY = "step_temporary"
    DEBUG_READBACK = "debug_readback"


@dataclass(frozen=True, slots=True)
class TensorSpec:
    dtype: str
    shape: tuple[int | str, ...]

    def rank(self) -> int:
        return len(self.shape)


@dataclass(frozen=True, slots=True)
class TensorLayout:
    name: str = "row_major"
    params: Mapping[str, int | str] | None = None

    @staticmethod
    def row_major() -> TensorLayout:
        return TensorLayout()


ROW_MAJOR_LAYOUT = TensorLayout.row_major()


@dataclass(frozen=True, slots=True)
class BufferSlice:
    allocation_id: str
    offset: int
    nbytes: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError(f"BufferSlice offset must be non-negative, got {self.offset}")
        if self.nbytes <= 0:
            raise ValueError(f"BufferSlice nbytes must be positive, got {self.nbytes}")


@dataclass(frozen=True, slots=True)
class WeightSource:
    key: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class ComparePolicy:
    kind: str = "tensor"
    rtol: float = 1e-3
    atol: float = 1e-3
    max_abs: float | None = None


@dataclass(frozen=True, slots=True)
class ReferenceRule:
    source: str
    selector: str | None = None


@dataclass(frozen=True, slots=True)
class LogicalTensor:
    name: str
    spec: TensorSpec
    layout: TensorLayout = ROW_MAJOR_LAYOUT
    role: TensorRole = TensorRole.ACTIVATION
    memory: MemoryPolicy = MemoryPolicy.FRAME_WORKSPACE
    storage: BufferSlice | None = None
    source: WeightSource | None = None
    ref: ReferenceRule | None = None
    compare: ComparePolicy | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LogicalTensor name must be non-empty")
        if self.spec.rank() == 0 and self.role is not TensorRole.TOKEN:
            raise ValueError(f"{self.name} must have at least one dimension")

    @property
    def shape(self) -> tuple[int | str, ...]:
        return self.spec.shape

    @property
    def dtype(self) -> str:
        return self.spec.dtype

    def bind(self, storage: BufferSlice) -> LogicalTensor:
        return replace(self, storage=storage)

    def with_name(self, name: str) -> LogicalTensor:
        return replace(self, name=name)

    def with_layout(self, layout: TensorLayout) -> LogicalTensor:
        return replace(self, layout=layout)

    def view_as(
        self,
        name: str,
        *,
        spec: TensorSpec,
        layout: TensorLayout | None = None,
    ) -> LogicalTensor:
        return LogicalTensor(
            name=name,
            spec=spec,
            layout=self.layout if layout is None else layout,
            role=self.role,
            memory=self.memory,
            storage=self.storage,
            source=self.source,
            ref=self.ref,
            compare=self.compare,
        )


def input_tensor(name: str, *, dtype: str, shape: tuple[int | str, ...]) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.INPUT,
        memory=MemoryPolicy.HOST_VISIBLE_INPUT,
    )


def output_tensor(name: str, *, dtype: str, shape: tuple[int | str, ...]) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.OUTPUT,
        memory=MemoryPolicy.HOST_VISIBLE_OUTPUT,
    )


def weight_tensor(
    name: str,
    *,
    dtype: str,
    shape: tuple[int | str, ...],
    source_key: str,
    source_dtype: str | None = None,
    source_shape: tuple[int, ...] | None = None,
    layout: TensorLayout | None = None,
) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        layout=ROW_MAJOR_LAYOUT if layout is None else layout,
        role=TensorRole.WEIGHT,
        memory=MemoryPolicy.DEVICE_LOCAL,
        source=WeightSource(key=source_key, dtype=source_dtype, shape=source_shape),
    )


def activation_tensor(
    name: str,
    *,
    dtype: str,
    shape: tuple[int | str, ...],
    role: TensorRole = TensorRole.ACTIVATION,
    memory: MemoryPolicy = MemoryPolicy.FRAME_WORKSPACE,
    ref: ReferenceRule | None = None,
    compare: ComparePolicy | None = None,
) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=role,
        memory=memory,
        ref=ref,
        compare=compare,
    )


def normalize_shape(shape: tuple[Any, ...]) -> tuple[int | str, ...]:
    normalized: list[int | str] = []
    for dim in shape:
        if isinstance(dim, int) or (isinstance(dim, str) and dim):
            normalized.append(dim)
        else:
            raise TypeError(f"Invalid tensor dimension {dim!r}")
    return tuple(normalized)
