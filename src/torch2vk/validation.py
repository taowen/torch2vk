"""Validation helpers for logical tensors and PyTorch-like references."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .logical import ComparePolicy, LogicalTensor
from .schema import BoundaryRule
from .shader import DispatchRecord


@dataclass(frozen=True, slots=True)
class ArtifactDifference:
    tensor: str
    reason: str


@dataclass(frozen=True, slots=True)
class WriterDrilldown:
    dispatch_index: int
    shader: str
    tensor: str
    matching_inputs: tuple[str, ...]
    divergent_inputs: tuple[ArtifactDifference, ...]
    unavailable_inputs: tuple[ArtifactDifference, ...]
    divergent_output: ArtifactDifference | None


@dataclass(frozen=True, slots=True)
class BoundaryMismatch:
    boundary: str
    tensor: str
    reason: str
    writer: WriterDrilldown | None

    def message(self) -> str:
        lines = [
            f"first mismatch: {self.tensor}",
            f"boundary: {self.boundary}",
            f"reason: {self.reason}",
        ]
        if self.writer is not None:
            lines.extend(
                (
                    f"writer shader: {self.writer.shader}",
                    f"writer dispatch: {self.writer.dispatch_index}",
                    "matching inputs: " + ", ".join(self.writer.matching_inputs),
                )
            )
            if self.writer.divergent_inputs:
                inputs = ", ".join(diff.tensor for diff in self.writer.divergent_inputs)
                lines.append(f"divergent inputs: {inputs}")
            if self.writer.unavailable_inputs:
                inputs = ", ".join(diff.tensor for diff in self.writer.unavailable_inputs)
                lines.append(f"unavailable inputs: {inputs}")
            if self.writer.divergent_output is not None:
                lines.append(f"divergent output: {self.writer.divergent_output.tensor}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class BoundaryCompareReport:
    mismatch: BoundaryMismatch | None

    @property
    def ok(self) -> bool:
        return self.mismatch is None

    def raise_for_mismatch(self) -> None:
        if self.mismatch is None:
            return
        raise AssertionError(self.mismatch.message())


@dataclass(frozen=True, slots=True)
class DebugReadbackPlan:
    tensor_names: tuple[str, ...]

    def __contains__(self, tensor_name: str) -> bool:
        return tensor_name in self.tensor_names


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


def compare_declared_boundaries(
    boundaries: Sequence[BoundaryRule],
    *,
    dispatch_records: Sequence[DispatchRecord],
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> BoundaryCompareReport:
    writers = _writer_by_tensor(dispatch_records)
    for boundary in sorted(boundaries, key=lambda item: item.order):
        for tensor in (*boundary.tensors, *boundary.tokens):
            policy = tensor.compare if tensor.compare is not None else boundary.compare
            difference = artifact_difference(
                tensor.name,
                reference=reference,
                candidate=candidate,
                policy=policy,
            )
            if difference is None:
                continue
            writer = writers.get(tensor.name)
            return BoundaryCompareReport(
                BoundaryMismatch(
                    boundary=boundary.name,
                    tensor=tensor.name,
                    reason=difference.reason,
                    writer=None
                    if writer is None
                    else drilldown_writer(
                        writer,
                        reference=reference,
                        candidate=candidate,
                        policy=policy,
                    ),
                )
            )
    return BoundaryCompareReport(None)


def debug_readback_plan(
    boundaries: Sequence[BoundaryRule],
    *,
    dispatch_records: Sequence[DispatchRecord],
) -> DebugReadbackPlan:
    writers = _writer_by_tensor(dispatch_records)
    ordered: list[str] = []
    seen: set[str] = set()
    for boundary in sorted(boundaries, key=lambda item: item.order):
        boundary_tensors = (*boundary.tensors, *boundary.tokens)
        _append_unique(ordered, seen, (tensor.name for tensor in boundary_tensors))
        if boundary.readback == "none":
            continue
        for tensor in boundary_tensors:
            writer = writers.get(tensor.name)
            if writer is None:
                continue
            if boundary.readback == "writer-output":
                _append_unique(ordered, seen, writer.writes.values())
            elif boundary.readback == "writer-io":
                _append_unique(ordered, seen, writer.reads.values())
                _append_unique(ordered, seen, writer.writes.values())
            else:
                raise ValueError(f"Unknown boundary readback policy {boundary.readback!r}")
    return DebugReadbackPlan(tuple(ordered))


def drilldown_writer(
    writer: DispatchRecord,
    *,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: ComparePolicy,
) -> WriterDrilldown:
    matching_inputs: list[str] = []
    divergent_inputs: list[ArtifactDifference] = []
    unavailable_inputs: list[ArtifactDifference] = []
    for tensor_name in writer.reads.values():
        difference = artifact_difference(
            tensor_name,
            reference=reference,
            candidate=candidate,
            policy=policy,
        )
        if difference is None:
            matching_inputs.append(tensor_name)
        elif difference.reason.startswith("missing "):
            unavailable_inputs.append(difference)
        else:
            divergent_inputs.append(difference)

    divergent_output: ArtifactDifference | None = None
    for tensor_name in writer.writes.values():
        difference = artifact_difference(
            tensor_name,
            reference=reference,
            candidate=candidate,
            policy=policy,
        )
        if difference is not None:
            divergent_output = difference
            break
    first_output = next(iter(writer.writes.values()), "")
    return WriterDrilldown(
        dispatch_index=writer.index,
        shader=writer.shader,
        tensor=divergent_output.tensor if divergent_output is not None else first_output,
        matching_inputs=tuple(matching_inputs),
        divergent_inputs=tuple(divergent_inputs),
        unavailable_inputs=tuple(unavailable_inputs),
        divergent_output=divergent_output,
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


def _writer_by_tensor(records: Sequence[DispatchRecord]) -> dict[str, DispatchRecord]:
    writers: dict[str, DispatchRecord] = {}
    for record in records:
        for tensor_name in record.writes.values():
            writers[tensor_name] = record
    return writers


def _append_unique(
    values: list[str],
    seen: set[str],
    names: Iterable[str],
) -> None:
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        values.append(name)


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
