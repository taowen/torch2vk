"""Model schema declarations shared by weights, execution, and debug."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .logical import (
    ROW_MAJOR_LAYOUT,
    ComparePolicy,
    LogicalTensor,
    ReferenceRule,
    TensorLayout,
    TensorRole,
    TensorSpec,
    weight_tensor,
)

DEFAULT_COMPARE_POLICY = ComparePolicy()


@dataclass(frozen=True, slots=True)
class StepScope:
    name: str
    prefix: str

    def key_prefix(self, step: int) -> str:
        return self.prefix.format(step=step)


@dataclass(frozen=True, slots=True)
class BoundaryRule:
    name: str
    phase: str
    order: int
    tensors: tuple[LogicalTensor, ...] = ()
    tokens: tuple[LogicalTensor, ...] = ()
    compare: ComparePolicy = DEFAULT_COMPARE_POLICY
    checkpoint: LogicalTensor | None = None
    readback: Literal["none", "writer-output", "writer-io"] = "none"


@dataclass(frozen=True, slots=True)
class ModelSchema:
    model: str
    weights: tuple[LogicalTensor, ...] = ()
    boundaries: tuple[BoundaryRule, ...] = ()

    def ordered_boundaries(self) -> tuple[BoundaryRule, ...]:
        return tuple(sorted(self.boundaries, key=lambda item: item.order))

    def weight_map(self) -> dict[str, LogicalTensor]:
        return {weight.name: weight for weight in self.weights}


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


def token(
    name: str,
    *,
    ref_source: str | None = None,
    step_scope: StepScope | None = None,
) -> LogicalTensor:
    full_name = name if step_scope is None else f"{step_scope.prefix}.{name}"
    return LogicalTensor(
        name=full_name,
        spec=TensorSpec(dtype="int32", shape=()),
        role=TensorRole.TOKEN,
        ref=None if ref_source is None else ReferenceRule(source=ref_source),
        compare=ComparePolicy(kind="token"),
    )
