"""Readback helpers for Vulkan tensor artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from .logical import LogicalTensor
from .vulkan_backend import VulkanBuffer
from .vulkan_runner import LogicalTensorLookup, read_bound_tensor_bytes

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "f16": torch.float16,
    "float16": torch.float16,
    "f32": torch.float32,
    "float32": torch.float32,
    "i32": torch.int32,
    "int32": torch.int32,
    "i64": torch.int64,
    "int64": torch.int64,
}


def read_bound_tensor_artifacts(
    tensors: LogicalTensorLookup,
    allocations: Mapping[str, VulkanBuffer],
    *,
    names: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    selected_names = tuple(tensors) if names is None else tuple(names)
    artifacts: dict[str, torch.Tensor] = {}
    for name in selected_names:
        try:
            tensor = _first_tensor(tensors[name])
        except KeyError as exc:
            raise KeyError(f"Missing LogicalTensor {name}") from exc
        artifacts[name] = bound_tensor_to_torch(tensor, allocations)
    return artifacts


def bound_tensor_to_torch(
    tensor: LogicalTensor,
    allocations: Mapping[str, VulkanBuffer],
) -> torch.Tensor:
    dtype = _torch_dtype(tensor)
    shape = _concrete_shape(tensor)
    data = bytearray(read_bound_tensor_bytes(tensor, allocations))
    return torch.frombuffer(data, dtype=dtype).reshape(shape).clone()


def _first_tensor(value: LogicalTensor | tuple[LogicalTensor, ...]) -> LogicalTensor:
    if isinstance(value, LogicalTensor):
        return value
    if not value:
        raise ValueError("LogicalTensor tuple must not be empty")
    return value[0]


def _torch_dtype(tensor: LogicalTensor) -> torch.dtype:
    normalized = tensor.dtype.lower()
    try:
        return _TORCH_DTYPES[normalized]
    except KeyError as exc:
        raise ValueError(f"{tensor.name} has unsupported artifact dtype {tensor.dtype!r}") from exc


def _concrete_shape(tensor: LogicalTensor) -> tuple[int, ...]:
    if any(not isinstance(dim, int) for dim in tensor.shape):
        raise ValueError(f"{tensor.name} has unresolved symbolic shape {tensor.shape}")
    return tuple(int(dim) for dim in tensor.shape)
