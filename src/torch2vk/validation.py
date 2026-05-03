"""Validation helpers for logical tensors and PyTorch-like references."""

from __future__ import annotations

from typing import Any

from .logical import LogicalTensor


def validate_tensor_matches_reference(tensor: LogicalTensor, reference: Any) -> None:
    dtype = getattr(reference, "dtype", None)
    shape = getattr(reference, "shape", None)
    if shape is None:
        raise TypeError(f"Reference for {tensor.name} has no shape")
    actual_shape = tuple(int(dim) for dim in shape)
    expected_shape = tensor.shape
    if any(not isinstance(dim, int) for dim in expected_shape):
        raise ValueError(f"{tensor.name} has unresolved symbolic shape {expected_shape}")
    if actual_shape != expected_shape:
        raise ValueError(
            f"{tensor.name} expected PyTorch shape {expected_shape}, got {actual_shape}"
        )
    if dtype is not None and not _dtype_matches(tensor.dtype, str(dtype)):
        raise ValueError(
            f"{tensor.name} expected PyTorch dtype compatible with {tensor.dtype}, got {dtype}"
        )


def _dtype_matches(logical_dtype: str, reference_dtype: str) -> bool:
    normalized = reference_dtype.replace("torch.", "").lower()
    aliases = {
        "bf16": {"bfloat16", "bf16"},
        "bfloat16": {"bfloat16", "bf16"},
        "f16": {"float16", "half", "f16"},
        "float16": {"float16", "half", "f16"},
        "f32": {"float32", "float", "f32"},
        "float32": {"float32", "float", "f32"},
        "i32": {"int32", "i32"},
        "int32": {"int32", "i32"},
    }
    return normalized in aliases.get(logical_dtype.lower(), {logical_dtype.lower()})
