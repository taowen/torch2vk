"""Frame-local candidate/reference tensor comparison."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Protocol, TypeGuard

import numpy as np

from torch2vk.runtime.logical import ComparePolicy, LogicalTensor

_TOP_ERROR_COUNT = 8


class _TorchTensorLike(Protocol):
    def detach(self) -> "_TorchTensorLike": ...

    def cpu(self) -> "_TorchTensorLike": ...

    @property
    def dtype(self) -> object: ...

    def float(self) -> "_TorchTensorLike": ...

    def numpy(self) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class TensorStats:
    min: float | None
    max: float | None
    mean: float | None
    nan_count: int
    inf_count: int


@dataclass(frozen=True, slots=True)
class MismatchEntry:
    index: tuple[int, ...]
    candidate: int | float | str
    expected: int | float | str
    abs_error: float
    rel_error: float


@dataclass(frozen=True, slots=True)
class TensorCompareResult:
    tensor: LogicalTensor
    frame: str
    artifact_key: str
    candidate_shape: tuple[int, ...]
    expected_shape: tuple[int, ...]
    candidate_dtype: str
    expected_dtype: str
    candidate_layout: str
    expected_layout: str
    max_abs: float
    max_rel: float
    passed: bool
    failure_reason: str | None = None
    first_mismatch_index: tuple[int, ...] | None = None
    candidate_value: int | float | str | None = None
    expected_value: int | float | str | None = None
    candidate_stats: TensorStats | None = None
    expected_stats: TensorStats | None = None
    top_errors: tuple[MismatchEntry, ...] = ()
    nearest_upstream_artifact_key: str | None = None
    drilldown_classification: str | None = None
    drilldown_artifact_path: str | None = None
    writer_input_artifact_paths: tuple[str, ...] = ()
    writer_output_artifact_paths: tuple[str, ...] = ()
    candidate_artifact_path: str | None = None
    expected_artifact_path: str | None = None
    summary_artifact_path: str | None = None


class CompareAssertionError(AssertionError):
    def __init__(self, *, policy: ComparePolicy, result: TensorCompareResult) -> None:
        self.policy = policy
        self.result = result
        super().__init__(_compare_error_message(policy=policy, result=result))


def compare_arrays(
    *,
    tensor: LogicalTensor,
    frame: str,
    candidate: object,
    expected: object,
    artifact_dir: str | Path | None = None,
    nearest_upstream_artifact_key: str | None = None,
) -> TensorCompareResult:
    if tensor.compare is None:
        raise ValueError(f"{tensor.name} has no ComparePolicy")
    policy = tensor.compare
    candidate_array = _as_numpy(candidate)
    expected_array = _as_numpy(expected)
    artifact_key = f"{frame}/{tensor.name}"
    if candidate_array.shape != expected_array.shape:
        result = _result(
            tensor=tensor,
            frame=frame,
            artifact_key=artifact_key,
            candidate_array=candidate_array,
            expected_array=expected_array,
            max_abs=float("nan"),
            max_rel=float("nan"),
            passed=False,
            failure_reason="shape_mismatch",
            nearest_upstream_artifact_key=nearest_upstream_artifact_key,
        )
        result = _dump_mismatch_artifacts(result, candidate_array, expected_array, artifact_dir)
        raise CompareAssertionError(policy=policy, result=result)

    if policy.kind == "token":
        passed = bool(np.array_equal(candidate_array, expected_array))
        abs_diff = np.abs(candidate_array.astype(np.float64) - expected_array.astype(np.float64))
        max_abs = float(abs_diff.max(initial=0.0))
        max_rel = 0.0
        first_mismatch_index, top_errors = _token_mismatches(
            candidate_array=candidate_array,
            expected_array=expected_array,
            abs_diff=abs_diff,
        )
    else:
        candidate_f64 = candidate_array.astype(np.float64, copy=False)
        expected_f64 = expected_array.astype(np.float64, copy=False)
        diff = np.abs(candidate_f64 - expected_f64)
        max_abs = _nan_safe_max(diff)
        denom = np.maximum(np.abs(expected_f64), np.finfo(np.float64).tiny)
        rel_diff = diff / denom
        max_rel = _nan_safe_max(rel_diff)
        tolerance = policy.atol + policy.rtol * np.abs(expected_f64)
        passed = bool(np.all(diff <= tolerance))
        if policy.max_abs is not None:
            passed = passed and max_abs <= policy.max_abs
        first_mismatch_index, top_errors = _tensor_mismatches(
            candidate_array=candidate_array,
            expected_array=expected_array,
            diff=diff,
            rel_diff=rel_diff,
            tolerance=tolerance,
            max_abs=max_abs,
            failed=not passed,
        )

    candidate_value: int | float | str | None = None
    expected_value: int | float | str | None = None
    if first_mismatch_index is not None:
        candidate_value = _scalar_value(candidate_array[first_mismatch_index])
        expected_value = _scalar_value(expected_array[first_mismatch_index])
    result = TensorCompareResult(
        tensor=tensor,
        frame=frame,
        artifact_key=artifact_key,
        candidate_shape=tuple(int(dim) for dim in candidate_array.shape),
        expected_shape=tuple(int(dim) for dim in expected_array.shape),
        candidate_dtype=str(candidate_array.dtype),
        expected_dtype=str(expected_array.dtype),
        candidate_layout=repr(tensor.layout),
        expected_layout="<pytorch>",
        max_abs=max_abs,
        max_rel=max_rel,
        passed=passed,
        failure_reason=None if passed else "value_mismatch",
        first_mismatch_index=first_mismatch_index,
        candidate_value=candidate_value,
        expected_value=expected_value,
        candidate_stats=_stats(candidate_array),
        expected_stats=_stats(expected_array),
        top_errors=top_errors,
        nearest_upstream_artifact_key=nearest_upstream_artifact_key,
    )
    if not passed:
        result = _dump_mismatch_artifacts(result, candidate_array, expected_array, artifact_dir)
        raise CompareAssertionError(policy=policy, result=result)
    return result


def normalize_reference_outputs(outputs: Mapping[object, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in outputs.items():
        if isinstance(key, LogicalTensor):
            normalized[key.name] = value
        elif isinstance(key, str):
            normalized[key] = value
        else:
            raise TypeError(
                f"Reference output key must be LogicalTensor or str, got {type(key).__name__}"
            )
    return normalized


def as_numpy_array(value: object) -> np.ndarray:
    return _as_numpy(value)


def write_compare_summary(result: TensorCompareResult) -> None:
    if result.summary_artifact_path is None:
        return
    summary_path = Path(result.summary_artifact_path)
    summary = _serializable_result(result)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if _is_torch_tensor(value):
        tensor = value.detach().cpu()
        if str(tensor.dtype) in {"torch.bfloat16", "torch.float16"}:
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def _is_torch_tensor(value: object) -> TypeGuard[_TorchTensorLike]:
    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy")


def _result(
    *,
    tensor: LogicalTensor,
    frame: str,
    artifact_key: str,
    candidate_array: np.ndarray,
    expected_array: np.ndarray,
    max_abs: float,
    max_rel: float,
    passed: bool,
    failure_reason: str | None,
    nearest_upstream_artifact_key: str | None,
) -> TensorCompareResult:
    return TensorCompareResult(
        tensor=tensor,
        frame=frame,
        artifact_key=artifact_key,
        candidate_shape=tuple(int(dim) for dim in candidate_array.shape),
        expected_shape=tuple(int(dim) for dim in expected_array.shape),
        candidate_dtype=str(candidate_array.dtype),
        expected_dtype=str(expected_array.dtype),
        candidate_layout=repr(tensor.layout),
        expected_layout="<pytorch>",
        max_abs=max_abs,
        max_rel=max_rel,
        passed=passed,
        failure_reason=failure_reason,
        candidate_stats=_stats(candidate_array),
        expected_stats=_stats(expected_array),
        nearest_upstream_artifact_key=nearest_upstream_artifact_key,
    )


def _compare_error_message(*, policy: ComparePolicy, result: TensorCompareResult) -> str:
    writer = result.tensor.writer
    writer_text = "<unknown>" if writer is None else f"{writer.shader}#{writer.dispatch_index}"
    lines = [
        "Tensor compare failed:",
        f"  frame: {result.frame}",
        f"  artifact: {result.artifact_key}",
        f"  tensor: {result.tensor.name}",
        f"  writer: {writer_text}",
        f"  reason: {result.failure_reason}",
        (
            "  candidate: "
            f"shape={result.candidate_shape}, dtype={result.candidate_dtype}, "
            f"layout={result.candidate_layout}"
        ),
        (
            "  expected: "
            f"shape={result.expected_shape}, dtype={result.expected_dtype}, "
            f"layout={result.expected_layout}"
        ),
        (
            "  policy: "
            f"{policy.kind}(rtol={policy.rtol}, atol={policy.atol}, max_abs_limit={policy.max_abs})"
        ),
        f"  max_abs: {_format_float(result.max_abs)}",
        f"  max_rel: {_format_float(result.max_rel)}",
    ]
    if result.first_mismatch_index is not None:
        lines.append(
            "  first_mismatch: "
            f"index={result.first_mismatch_index}, "
            f"candidate={result.candidate_value}, expected={result.expected_value}"
        )
    if result.top_errors:
        top_text = "; ".join(
            (
                f"{entry.index}: cand={entry.candidate}, exp={entry.expected}, "
                f"abs={_format_float(entry.abs_error)}, rel={_format_float(entry.rel_error)}"
            )
            for entry in result.top_errors[:3]
        )
        lines.append(f"  top_errors: {top_text}")
    if result.candidate_stats is not None:
        lines.append(f"  candidate_stats: {_format_stats(result.candidate_stats)}")
    if result.expected_stats is not None:
        lines.append(f"  expected_stats: {_format_stats(result.expected_stats)}")
    if result.nearest_upstream_artifact_key is not None:
        lines.append(f"  nearest_upstream_compared: {result.nearest_upstream_artifact_key}")
    if result.drilldown_classification is not None:
        lines.append(f"  drilldown_classification: {result.drilldown_classification}")
    if result.drilldown_artifact_path is not None:
        lines.append(f"  writer_io_drilldown: {result.drilldown_artifact_path}")
    if result.candidate_artifact_path is not None:
        lines.append(f"  candidate_dump: {result.candidate_artifact_path}")
    if result.expected_artifact_path is not None:
        lines.append(f"  expected_dump: {result.expected_artifact_path}")
    if result.summary_artifact_path is not None:
        lines.append(f"  summary_dump: {result.summary_artifact_path}")
    return "\n".join(lines)


def _dump_mismatch_artifacts(
    result: TensorCompareResult,
    candidate_array: np.ndarray,
    expected_array: np.ndarray,
    artifact_dir: str | Path | None,
) -> TensorCompareResult:
    if artifact_dir is None:
        return result
    root = (
        Path(artifact_dir)
        / "debug"
        / _safe_path_component(result.frame)
        / _safe_path_component(result.tensor.name)
    )
    root.mkdir(parents=True, exist_ok=True)
    candidate_path = root / "candidate.npy"
    expected_path = root / "expected.npy"
    summary_path = root / "summary.json"
    np.save(candidate_path, candidate_array)
    np.save(expected_path, expected_array)
    summary = _serializable_result(result)
    summary["candidate_artifact_path"] = str(candidate_path)
    summary["expected_artifact_path"] = str(expected_path)
    summary["summary_artifact_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return TensorCompareResult(
        tensor=result.tensor,
        frame=result.frame,
        artifact_key=result.artifact_key,
        candidate_shape=result.candidate_shape,
        expected_shape=result.expected_shape,
        candidate_dtype=result.candidate_dtype,
        expected_dtype=result.expected_dtype,
        candidate_layout=result.candidate_layout,
        expected_layout=result.expected_layout,
        max_abs=result.max_abs,
        max_rel=result.max_rel,
        passed=result.passed,
        failure_reason=result.failure_reason,
        first_mismatch_index=result.first_mismatch_index,
        candidate_value=result.candidate_value,
        expected_value=result.expected_value,
        candidate_stats=result.candidate_stats,
        expected_stats=result.expected_stats,
        top_errors=result.top_errors,
        nearest_upstream_artifact_key=result.nearest_upstream_artifact_key,
        drilldown_classification=result.drilldown_classification,
        drilldown_artifact_path=result.drilldown_artifact_path,
        writer_input_artifact_paths=result.writer_input_artifact_paths,
        writer_output_artifact_paths=result.writer_output_artifact_paths,
        candidate_artifact_path=str(candidate_path),
        expected_artifact_path=str(expected_path),
        summary_artifact_path=str(summary_path),
    )


def select_probe_value(value: object, *, index: int, selector: str | None) -> Any:
    selected = value
    if isinstance(selected, (tuple, list)):
        selected = selected[index]
    elif index != 0:
        raise IndexError(f"Probe index {index} requested for non-sequence value")
    if selector is None:
        return selected
    if isinstance(selected, Mapping):
        return selected[selector]
    return getattr(selected, selector)


def _tensor_mismatches(
    *,
    candidate_array: np.ndarray,
    expected_array: np.ndarray,
    diff: np.ndarray,
    rel_diff: np.ndarray,
    tolerance: np.ndarray,
    max_abs: float,
    failed: bool,
) -> tuple[tuple[int, ...] | None, tuple[MismatchEntry, ...]]:
    if not failed:
        return None, ()
    mismatch = ~(diff <= tolerance)
    if not bool(np.any(mismatch)) and not np.isnan(max_abs):
        mismatch = diff == max_abs
    first = _first_index(mismatch)
    return first, _top_errors(
        candidate_array=candidate_array,
        expected_array=expected_array,
        abs_diff=diff,
        rel_diff=rel_diff,
    )


def _token_mismatches(
    *,
    candidate_array: np.ndarray,
    expected_array: np.ndarray,
    abs_diff: np.ndarray,
) -> tuple[tuple[int, ...] | None, tuple[MismatchEntry, ...]]:
    mismatch = candidate_array != expected_array
    rel_diff = np.zeros_like(abs_diff, dtype=np.float64)
    first = _first_index(mismatch)
    return first, _top_errors(
        candidate_array=candidate_array,
        expected_array=expected_array,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
    )


def _first_index(mask: np.ndarray) -> tuple[int, ...] | None:
    flat_indices = np.flatnonzero(mask)
    if flat_indices.size == 0:
        return None
    return tuple(int(dim) for dim in np.unravel_index(int(flat_indices[0]), mask.shape))


def _top_errors(
    *,
    candidate_array: np.ndarray,
    expected_array: np.ndarray,
    abs_diff: np.ndarray,
    rel_diff: np.ndarray,
) -> tuple[MismatchEntry, ...]:
    if abs_diff.size == 0:
        return ()
    sortable = np.nan_to_num(abs_diff.reshape(-1), nan=np.inf, posinf=np.inf, neginf=np.inf)
    count = min(_TOP_ERROR_COUNT, sortable.size)
    if count == 0:
        return ()
    if count < sortable.size:
        candidate_indices = np.argpartition(sortable, -count)[-count:]
        ordered = candidate_indices[np.argsort(sortable[candidate_indices])[::-1]]
    else:
        ordered = np.argsort(sortable)[::-1]
    entries: list[MismatchEntry] = []
    for flat_index in ordered:
        index = tuple(int(dim) for dim in np.unravel_index(int(flat_index), abs_diff.shape))
        entries.append(
            MismatchEntry(
                index=index,
                candidate=_scalar_value(candidate_array[index]),
                expected=_scalar_value(expected_array[index]),
                abs_error=float(abs_diff[index]),
                rel_error=float(rel_diff[index]),
            )
        )
    return tuple(entries)


def _stats(array: np.ndarray) -> TensorStats | None:
    if array.size == 0:
        return TensorStats(min=None, max=None, mean=None, nan_count=0, inf_count=0)
    try:
        numeric = array.astype(np.float64, copy=False)
    except (TypeError, ValueError):
        return None
    finite = np.isfinite(numeric)
    if bool(np.any(finite)):
        finite_values = numeric[finite]
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        mean_value = float(finite_values.mean())
    else:
        min_value = None
        max_value = None
        mean_value = None
    return TensorStats(
        min=min_value,
        max=max_value,
        mean=mean_value,
        nan_count=int(np.isnan(numeric).sum()),
        inf_count=int(np.isinf(numeric).sum()),
    )


def _nan_safe_max(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0
    if bool(np.all(np.isnan(array))):
        return float("nan")
    return float(np.nanmax(array))


def _scalar_value(value: object) -> int | float | str:
    item = getattr(value, "item", None)
    scalar = item() if callable(item) else value
    if isinstance(scalar, bool):
        return int(scalar)
    if isinstance(scalar, int | np.integer):
        return int(scalar)
    if isinstance(scalar, float | np.floating):
        return float(scalar)
    return str(scalar)


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.8g}"


def _format_stats(stats: TensorStats) -> str:
    return (
        f"min={None if stats.min is None else _format_float(stats.min)}, "
        f"max={None if stats.max is None else _format_float(stats.max)}, "
        f"mean={None if stats.mean is None else _format_float(stats.mean)}, "
        f"nan={stats.nan_count}, inf={stats.inf_count}"
    )


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "artifact"


def _serializable_result(result: TensorCompareResult) -> dict[str, object]:
    writer = result.tensor.writer
    return {
        "frame": result.frame,
        "artifact_key": result.artifact_key,
        "tensor": result.tensor.name,
        "writer": None
        if writer is None
        else {
            "frame": writer.frame,
            "shader": writer.shader,
            "dispatch_index": writer.dispatch_index,
        },
        "candidate_shape": result.candidate_shape,
        "expected_shape": result.expected_shape,
        "candidate_dtype": result.candidate_dtype,
        "expected_dtype": result.expected_dtype,
        "candidate_layout": result.candidate_layout,
        "expected_layout": result.expected_layout,
        "max_abs": result.max_abs,
        "max_rel": result.max_rel,
        "passed": result.passed,
        "failure_reason": result.failure_reason,
        "first_mismatch_index": result.first_mismatch_index,
        "candidate_value": result.candidate_value,
        "expected_value": result.expected_value,
        "candidate_stats": None
        if result.candidate_stats is None
        else asdict(result.candidate_stats),
        "expected_stats": None if result.expected_stats is None else asdict(result.expected_stats),
        "top_errors": [asdict(entry) for entry in result.top_errors],
        "nearest_upstream_artifact_key": result.nearest_upstream_artifact_key,
        "drilldown_classification": result.drilldown_classification,
        "drilldown_artifact_path": result.drilldown_artifact_path,
        "writer_input_artifact_paths": result.writer_input_artifact_paths,
        "writer_output_artifact_paths": result.writer_output_artifact_paths,
        "candidate_artifact_path": result.candidate_artifact_path,
        "expected_artifact_path": result.expected_artifact_path,
        "summary_artifact_path": result.summary_artifact_path,
    }
