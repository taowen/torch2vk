"""Validation helpers for logical tensors and PyTorch-like references."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .logical import ComparePolicy, LogicalTensor
from .shader import DispatchRecord


@dataclass(frozen=True, slots=True)
class ArtifactDifference:
    tensor: str
    reason: str


@dataclass(frozen=True, slots=True)
class DispatchReadWriteIssue:
    dispatch_index: int
    shader: str
    field: str
    tensor: str


@dataclass(frozen=True, slots=True)
class DispatchReadWriteReport:
    issues: tuple[DispatchReadWriteIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def raise_for_issues(self) -> None:
        if not self.issues:
            return
        first = self.issues[0]
        raise ValueError(
            "Dispatch reads tensor without an earlier writer or initial declaration: "
            f"{first.shader}.{first.field} -> {first.tensor} at dispatch {first.dispatch_index}"
        )


def validate_dispatch_read_write_chain(
    dispatch_records: Sequence[DispatchRecord],
    *,
    initial_tensors: Sequence[LogicalTensor] = (),
) -> DispatchReadWriteReport:
    written = {tensor.name for tensor in initial_tensors}
    issues: list[DispatchReadWriteIssue] = []
    for record in dispatch_records:
        for field, tensor_name in record.reads.items():
            if tensor_name in written:
                continue
            issues.append(
                DispatchReadWriteIssue(
                    dispatch_index=record.index,
                    shader=record.shader,
                    field=field,
                    tensor=tensor_name,
                )
            )
        written.update(record.writes.values())
    return DispatchReadWriteReport(tuple(issues))


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


def artifact_difference(
    tensor_name: str,
    *,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: ComparePolicy,
) -> ArtifactDifference | None:
    if tensor_name not in reference:
        return ArtifactDifference(tensor=tensor_name, reason="missing PyTorch reference")
    if tensor_name not in candidate:
        return ArtifactDifference(tensor=tensor_name, reason="missing candidate artifact")
    reason = _value_difference(reference[tensor_name], candidate[tensor_name], policy=policy)
    if reason is None:
        return None
    return ArtifactDifference(tensor=tensor_name, reason=reason)


def _value_difference(reference: Any, candidate: Any, *, policy: ComparePolicy) -> str | None:
    ref_shape = _shape_of(reference)
    cand_shape = _shape_of(candidate)
    if ref_shape is not None and cand_shape is not None and ref_shape != cand_shape:
        return f"shape mismatch reference={ref_shape} candidate={cand_shape}"

    ref_dtype = getattr(reference, "dtype", None)
    cand_dtype = getattr(candidate, "dtype", None)
    if ref_dtype is not None and cand_dtype is not None and str(ref_dtype) != str(cand_dtype):
        return f"dtype mismatch reference={ref_dtype} candidate={cand_dtype}"

    if hasattr(reference, "detach") and hasattr(candidate, "detach"):
        return _torch_value_difference(reference, candidate, policy=policy)
    if reference != candidate:
        return "value mismatch"
    return None


def _torch_value_difference(reference: Any, candidate: Any, *, policy: ComparePolicy) -> str | None:
    ref = reference.detach().cpu()
    cand = candidate.detach().cpu()
    if policy.kind == "token":
        equal = ref.equal(cand)
    else:
        equal = ref.allclose(cand, rtol=policy.rtol, atol=policy.atol)
    if bool(equal):
        return None
    max_abs = (ref.float() - cand.float()).abs().max().item()
    if policy.max_abs is not None and max_abs <= policy.max_abs:
        return None
    return f"value mismatch max_abs={max_abs}"


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


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
