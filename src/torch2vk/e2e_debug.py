"""End-to-end debug boundary comparison and root-cause attribution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from .logical import ComparePolicy
from .validation import artifact_difference

type DispatchMap = Mapping[str, Sequence[Mapping[str, Any]]]
type Evidence = Mapping[str, Any]


def _empty_dispatch_map() -> DispatchMap:
    return {}


def _empty_tensors() -> Mapping[str, torch.Tensor]:
    return {}


def _empty_evidence() -> Evidence:
    return {}


@dataclass(frozen=True, slots=True)
class DebugBoundary:
    name: str
    order: int
    compare: ComparePolicy
    step_scoped: bool = True
    artifacts: tuple[str, ...] = ()

    def key(self, step: int | None) -> str:
        if not self.step_scoped:
            return self.name
        if step is None:
            raise ValueError(f"{self.name} requires a step")
        return f"step_{step:03d}.{self.name}"

    def artifact_keys(self, step: int | None) -> tuple[str, ...]:
        names = self.artifacts or (self.name,)
        if not self.step_scoped:
            return names
        if step is None:
            raise ValueError(f"{self.name} requires a step")
        return tuple(f"step_{step:03d}.{name}" for name in names)


@dataclass(frozen=True, slots=True)
class E2EDebugTrace:
    tensors: Mapping[str, torch.Tensor]
    tokens: Mapping[str, torch.Tensor] = field(default_factory=_empty_tensors)
    dispatches: DispatchMap = field(default_factory=_empty_dispatch_map)


@dataclass(frozen=True, slots=True)
class DrilldownResult:
    classification: str
    first_bad_dispatch: str | None
    evidence: Evidence = field(default_factory=_empty_evidence)


@dataclass(frozen=True, slots=True)
class RootCauseHop:
    step: int | None
    boundary: str
    classification: str
    dispatch: str | None
    evidence: Evidence = field(default_factory=_empty_evidence)


@dataclass(frozen=True, slots=True)
class E2EDebugReport:
    status: str
    first_bad_step: int | None
    first_bad_boundary: str | None
    first_bad_dispatch: str | None
    classification: str
    hops: tuple[RootCauseHop, ...] = ()

    def raise_for_mismatch(self) -> None:
        if self.status == "match":
            return
        raise AssertionError(
            "\n".join(
                (
                    "end-to-end debug mismatch",
                    f"status: {self.status}",
                    f"first_bad_step: {self.first_bad_step}",
                    f"first_bad_boundary: {self.first_bad_boundary}",
                    f"first_bad_dispatch: {self.first_bad_dispatch}",
                    f"classification: {self.classification}",
                )
            )
        )


@dataclass(frozen=True, slots=True)
class BoundaryCoverageReport:
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing


type DrilldownFn = Callable[[int | None, DebugBoundary], DrilldownResult]


def boundary_coverage(
    *,
    boundaries: Sequence[DebugBoundary],
    steps: int,
    trace: E2EDebugTrace,
) -> BoundaryCoverageReport:
    missing: list[str] = []
    for boundary in sorted(boundaries, key=lambda item: item.order):
        keys: tuple[str, ...]
        if boundary.step_scoped:
            keys = tuple(
                key
                for step in range(steps)
                for key in boundary.artifact_keys(step)
            )
        else:
            keys = boundary.artifact_keys(None)
        artifacts = _boundary_artifacts(boundary, trace)
        missing.extend(key for key in keys if key not in artifacts)
    return BoundaryCoverageReport(tuple(missing))


def trace_e2e_root_cause(
    *,
    boundaries: Sequence[DebugBoundary],
    steps: int,
    reference: E2EDebugTrace,
    candidate: E2EDebugTrace,
    drilldown: DrilldownFn,
    max_hops: int = 16,
) -> E2EDebugReport:
    ordered = tuple(sorted(boundaries, key=lambda boundary: boundary.order))
    first_step, first_boundary = _first_bad_boundary(
        boundaries=ordered,
        steps=steps,
        reference=reference,
        candidate=candidate,
    )
    if first_boundary is None:
        return E2EDebugReport(
            status="match",
            first_bad_step=None,
            first_bad_boundary=None,
            first_bad_dispatch=None,
            classification="match",
        )

    current_step = first_step
    current_boundary = first_boundary
    visited: set[tuple[int | None, str]] = set()
    hops: list[RootCauseHop] = []
    first_dispatch: str | None = None

    for _ in range(max_hops):
        key = (current_step, current_boundary.name)
        if key in visited:
            return _report(
                status="cycle_detected",
                first_step=first_step,
                first_boundary=first_boundary,
                first_dispatch=first_dispatch,
                classification="cycle_detected",
                hops=hops,
            )
        visited.add(key)
        result = drilldown(current_step, current_boundary)
        first_dispatch = first_dispatch or result.first_bad_dispatch
        hop = RootCauseHop(
            step=current_step,
            boundary=current_boundary.name,
            classification=result.classification,
            dispatch=result.first_bad_dispatch,
            evidence=result.evidence,
        )
        hops.append(hop)

        if result.classification == "input_ok_output_bad":
            return _report(
                status="root_cause_found",
                first_step=first_step,
                first_boundary=first_boundary,
                first_dispatch=result.first_bad_dispatch,
                classification=result.classification,
                hops=hops,
            )
        if result.classification != "input_bad_output_bad":
            return _report(
                status=result.classification,
                first_step=first_step,
                first_boundary=first_boundary,
                first_dispatch=first_dispatch,
                classification=result.classification,
                hops=hops,
            )
        previous = _previous_boundary(ordered, current_step, current_boundary)
        if previous is None:
            return _report(
                status="boundary_coverage_insufficient",
                first_step=first_step,
                first_boundary=first_boundary,
                first_dispatch=first_dispatch,
                classification="boundary_coverage_insufficient",
                hops=hops,
            )
        current_step, current_boundary = previous

    return _report(
        status="max_hops_exceeded",
        first_step=first_step,
        first_boundary=first_boundary,
        first_dispatch=first_dispatch,
        classification="max_hops_exceeded",
        hops=hops,
    )


def _first_bad_boundary(
    *,
    boundaries: Sequence[DebugBoundary],
    steps: int,
    reference: E2EDebugTrace,
    candidate: E2EDebugTrace,
) -> tuple[int | None, DebugBoundary | None]:
    for step in range(steps):
        for boundary in boundaries:
            if not boundary.step_scoped:
                continue
            if _boundary_differs(boundary, step, reference, candidate):
                return step, boundary
    for boundary in boundaries:
        if boundary.step_scoped:
            continue
        if _boundary_differs(boundary, None, reference, candidate):
            return None, boundary
    return None, None


def _boundary_differs(
    boundary: DebugBoundary,
    step: int | None,
    reference: E2EDebugTrace,
    candidate: E2EDebugTrace,
) -> bool:
    reference_artifacts, candidate_artifacts = _artifact_maps(boundary, reference, candidate)
    for artifact_key in boundary.artifact_keys(step):
        if (
            artifact_difference(
                artifact_key,
                reference=reference_artifacts,
                candidate=candidate_artifacts,
                policy=boundary.compare,
            )
            is not None
        ):
            return True
    return False


def _artifact_maps(
    boundary: DebugBoundary,
    reference: E2EDebugTrace,
    candidate: E2EDebugTrace,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    if boundary.compare.kind in {"token", "token_sequence"}:
        return (
            reference.tokens if reference.tokens else reference.tensors,
            candidate.tokens if candidate.tokens else candidate.tensors,
        )
    return reference.tensors, candidate.tensors


def _boundary_artifacts(
    boundary: DebugBoundary,
    trace: E2EDebugTrace,
) -> Mapping[str, torch.Tensor]:
    if boundary.compare.kind in {"token", "token_sequence"}:
        return trace.tokens if trace.tokens else trace.tensors
    return trace.tensors


def _previous_boundary(
    boundaries: Sequence[DebugBoundary],
    step: int | None,
    current: DebugBoundary,
) -> tuple[int | None, DebugBoundary] | None:
    same_step = [
        boundary
        for boundary in boundaries
        if boundary.step_scoped == current.step_scoped and boundary.order < current.order
    ]
    if same_step:
        return step, same_step[-1]
    if current.step_scoped and step is not None and step > 0:
        previous_step = [boundary for boundary in boundaries if boundary.step_scoped]
        if previous_step:
            return step - 1, previous_step[-1]
    return None


def _report(
    *,
    status: str,
    first_step: int | None,
    first_boundary: DebugBoundary,
    first_dispatch: str | None,
    classification: str,
    hops: Sequence[RootCauseHop],
) -> E2EDebugReport:
    return E2EDebugReport(
        status=status,
        first_bad_step=first_step,
        first_bad_boundary=first_boundary.name,
        first_bad_dispatch=first_dispatch,
        classification=classification,
        hops=tuple(hops),
    )
