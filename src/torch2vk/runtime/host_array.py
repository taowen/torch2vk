"""Host-side array validation for runtime inputs."""

from __future__ import annotations

import numpy as np

from torch2vk.runtime.logical import LogicalTensor
from torch2vk.vulkan.types import concrete_shape


def as_float16_array(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float16)


def as_float16_attention_mask(value: np.ndarray) -> np.ndarray:
    mask = np.empty(value.shape, dtype=np.float16)
    mask[value == 0] = np.float16(0.0)
    mask[value != 0] = np.float16(-np.inf)
    return np.ascontiguousarray(mask)


def prepare_host_array(
    tensor: LogicalTensor,
    value: object,
    *,
    context: str,
) -> np.ndarray:
    """Validate a host value against a LogicalTensor before upload."""
    array = np.asarray(value)
    expected_shape = concrete_shape(tensor.spec)
    if array.shape != expected_shape:
        raise ValueError(
            f"{tensor.name} {context} shape mismatch: "
            f"expected {expected_shape}, got {array.shape}"
        )

    if tensor.spec.dtype == "bool":
        if array.dtype != np.dtype(np.bool_):
            raise TypeError(
                f"{tensor.name} {context} dtype mismatch: expected bool, got {array.dtype}"
            )
        return np.ascontiguousarray(array.astype(np.uint32))

    try:
        expected_dtype = np.dtype(tensor.spec.dtype)
    except TypeError as exc:
        raise TypeError(
            f"{tensor.name} {context} cannot be uploaded from host with "
            f"dtype {tensor.spec.dtype!r}"
        ) from exc
    if array.dtype != expected_dtype:
        raise TypeError(
            f"{tensor.name} {context} dtype mismatch: "
            f"expected {expected_dtype}, got {array.dtype}"
        )
    return np.ascontiguousarray(array)
