"""Small helpers for model tensor declarations."""

from __future__ import annotations

from .logical import (
    ROW_MAJOR_LAYOUT,
    LogicalTensor,
    TensorLayout,
    weight_tensor,
)


def W(
    name: str,
    *,
    safetensor_key: str,
    dtype: str,
    shape: tuple[int, ...],
    layout: TensorLayout | None = None,
) -> LogicalTensor:
    return weight_tensor(
        name,
        dtype=dtype,
        shape=shape,
        source_key=safetensor_key,
        source_dtype=dtype,
        source_shape=shape,
        layout=ROW_MAJOR_LAYOUT if layout is None else layout,
    )
