"""Reference comparison helpers for RuntimeSession."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch2vk.runtime.compare import (
    CompareAssertionError,
    _DEFAULT_COMPARE_POLICY,
    compare_arrays,
    write_compare_summary,
)
from torch2vk.runtime.logical import ComparePolicy, LogicalTensor
from torch2vk.runtime.reference import ReferenceSpec

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


def compare_expected_with_spec(
    rt: RuntimeSession,
    *,
    name: str,
    tensors: object,
    spec: ReferenceSpec,
    expected: dict[str, object],
    policy: ComparePolicy | dict[str, ComparePolicy] = _DEFAULT_COMPARE_POLICY,
) -> None:
    """Compare Vulkan outputs against expected values computed by the caller."""
    for key, field_name in spec.output_bindings.items():
        if key not in expected:
            raise RuntimeError(
                f"compare_expected_with_spec {name!r}: output key {key!r} not in "
                f"expected results (available: {sorted(expected.keys())[:10]})"
            )
        tensor = _logical_tensor_path(tensors, field_name)
        candidate = rt.readback(tensor)
        try:
            result = compare_arrays(
                tensor=tensor,
                frame=name,
                candidate=candidate,
                expected=expected[key],
                policy=_policy_for_key(policy, key),
                artifact_dir=rt.artifact_dir,
            )
        except CompareAssertionError as exc:
            write_compare_summary(exc.result)
            rt._compare_results.append(exc.result)
            raise
        rt._compare_results.append(result)


def _logical_tensor_path(tensors: object, field_path: str) -> LogicalTensor:
    value: object = tensors
    for segment in field_path.split("."):
        if isinstance(value, (tuple, list)) and segment.isdecimal():
            value = value[int(segment)]
        else:
            value = getattr(value, segment)
    if not isinstance(value, LogicalTensor):
        raise TypeError(f"{type(tensors).__name__}.{field_path} is not a LogicalTensor")
    return value


def _policy_for_key(
    policy: ComparePolicy | dict[str, ComparePolicy],
    key: str,
) -> ComparePolicy:
    if isinstance(policy, dict):
        return policy[key]
    return policy
