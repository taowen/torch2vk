"""Frame-local candidate/reference tensor comparison."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeGuard

import numpy as np

from torch2vk.runtime.logical import ComparePolicy, LogicalTensor


class _TorchTensorLike(Protocol):
    def detach(self) -> "_TorchTensorLike": ...

    def cpu(self) -> "_TorchTensorLike": ...

    @property
    def dtype(self) -> object: ...

    def float(self) -> "_TorchTensorLike": ...

    def numpy(self) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class TensorCompareResult:
    tensor: LogicalTensor
    frame: str
    artifact_key: str
    max_abs: float
    max_rel: float
    passed: bool


def compare_arrays(
    *,
    tensor: LogicalTensor,
    frame: str,
    candidate: object,
    expected: object,
) -> TensorCompareResult:
    if tensor.compare is None:
        raise ValueError(f"{tensor.name} has no ComparePolicy")
    policy = tensor.compare
    candidate_array = _as_numpy(candidate)
    expected_array = _as_numpy(expected)
    if candidate_array.shape != expected_array.shape:
        raise AssertionError(
            f"{tensor.name} compare shape mismatch in {frame}: "
            f"candidate {candidate_array.shape}, expected {expected_array.shape}"
        )
    if policy.kind == "token":
        passed = bool(np.array_equal(candidate_array, expected_array))
        abs_diff = np.abs(candidate_array.astype(np.float64) - expected_array.astype(np.float64))
        max_abs = float(abs_diff.max(initial=0.0))
        max_rel = 0.0
    else:
        candidate_f64 = candidate_array.astype(np.float64, copy=False)
        expected_f64 = expected_array.astype(np.float64, copy=False)
        diff = np.abs(candidate_f64 - expected_f64)
        max_abs = float(diff.max(initial=0.0))
        denom = np.maximum(np.abs(expected_f64), np.finfo(np.float64).tiny)
        max_rel = float((diff / denom).max(initial=0.0))
        passed = bool(np.all(diff <= policy.atol + policy.rtol * np.abs(expected_f64)))
        if policy.max_abs is not None:
            passed = passed and max_abs <= policy.max_abs
    result = TensorCompareResult(
        tensor=tensor,
        frame=frame,
        artifact_key=f"{frame}/{tensor.name}",
        max_abs=max_abs,
        max_rel=max_rel,
        passed=passed,
    )
    if not passed:
        raise _compare_error(policy=policy, result=result)
    return result


def normalize_reference_outputs(outputs: Mapping[object, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in outputs.items():
        if isinstance(key, LogicalTensor):
            normalized[key.name] = value
        elif isinstance(key, str):
            normalized[key] = value
        else:
            raise TypeError(f"Reference output key must be LogicalTensor or str, got {type(key).__name__}")
    return normalized


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


def _compare_error(*, policy: ComparePolicy, result: TensorCompareResult) -> AssertionError:
    writer = result.tensor.writer
    writer_text = "<unknown>" if writer is None else f"{writer.shader}#{writer.dispatch_index}"
    return AssertionError(
        f"Tensor compare failed: artifact={result.artifact_key}, tensor={result.tensor.name}, "
        f"writer={writer_text}, policy={policy.kind}(rtol={policy.rtol}, atol={policy.atol}, "
        f"max_abs_limit={policy.max_abs}), max_abs={result.max_abs:.8g}, max_rel={result.max_rel:.8g}"
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
