"""Small physical storage planning helpers for logical tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .logical import BufferSlice, LogicalTensor

DTYPE_NBYTES: Mapping[str, int] = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "int32": 4,
    "int64": 8,
}


@dataclass(frozen=True, slots=True)
class StoragePlan:
    slices: Mapping[str, BufferSlice]

    def bind(self, tensor: LogicalTensor) -> LogicalTensor:
        storage = self.slices.get(tensor.name)
        if storage is None:
            raise KeyError(f"Storage plan has no slice for {tensor.name}")
        return tensor.bind(storage)


def plan_storage(
    tensors: Sequence[LogicalTensor],
    *,
    allocation_id: str,
    alignment: int = 256,
) -> StoragePlan:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    required: dict[str, int] = {}
    for tensor in tensors:
        required[tensor.name] = max(required.get(tensor.name, 0), tensor_nbytes(tensor))

    offset = 0
    slices: dict[str, BufferSlice] = {}
    for name, nbytes in sorted(required.items()):
        offset = align_up(offset, alignment)
        slices[name] = BufferSlice(allocation_id=allocation_id, offset=offset, nbytes=nbytes)
        offset += nbytes
    return StoragePlan(slices)


def bind_storage(tensors: Sequence[LogicalTensor], plan: StoragePlan) -> tuple[LogicalTensor, ...]:
    return tuple(plan.bind(tensor) for tensor in tensors)


def tensor_nbytes(tensor: LogicalTensor) -> int:
    width = DTYPE_NBYTES.get(tensor.dtype)
    if width is None:
        raise ValueError(f"{tensor.name} has unsupported dtype {tensor.dtype!r}")
    elements = 1
    for dim in tensor.shape:
        if not isinstance(dim, int):
            raise TypeError(f"{tensor.name} has symbolic shape {tensor.shape}")
        elements *= dim
    return elements * width


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
