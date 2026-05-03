"""Reference artifacts sourced from an external end-to-end trace."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from .logical import LogicalTensor
from .pytorch import ArtifactCache, TransformFn, required_probe_names

type ReferenceTraceCaptureFn = Callable[[Mapping[str, Any]], "ReferenceTrace"]
type TimelineEvent = Mapping[str, Any]


def _empty_tensors() -> Mapping[str, torch.Tensor]:
    return {}


def _empty_metadata() -> Mapping[str, Any]:
    return {}


def _empty_timeline() -> Sequence[TimelineEvent]:
    return ()


@dataclass(frozen=True, slots=True)
class ReferenceTrace:
    tensors: Mapping[str, torch.Tensor] = field(default_factory=_empty_tensors)
    tokens: Mapping[str, torch.Tensor] = field(default_factory=_empty_tensors)
    timeline: Sequence[TimelineEvent] = field(default_factory=_empty_timeline)
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class TraceReferenceProvider:
    capture: ReferenceTraceCaptureFn
    provider_id: str

    def ensure(
        self,
        *,
        tensors: tuple[LogicalTensor, ...],
        inputs: Mapping[str, Any],
        cache: ArtifactCache,
        transforms: Mapping[str, TransformFn] | None = None,
        extra_fingerprint: Mapping[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        del transforms
        required = required_probe_names(tensors)
        fingerprint = _trace_reference_fingerprint(
            provider_id=self.provider_id,
            tensors=tensors,
            inputs=inputs,
            required=required,
            extra=extra_fingerprint or {},
        )
        cached = cache.load(fingerprint)
        if _cached_trace_complete(cached, required=required):
            return dict(cached)

        trace = self.capture(inputs)
        artifacts = _artifacts_from_trace(tensors=tensors, required=required, trace=trace)
        cache.store(fingerprint, artifacts)
        return artifacts


def _artifacts_from_trace(
    *,
    tensors: tuple[LogicalTensor, ...],
    required: tuple[str, ...],
    trace: ReferenceTrace,
) -> dict[str, torch.Tensor]:
    by_name = {tensor.name: tensor for tensor in tensors}
    artifacts: dict[str, torch.Tensor] = {}
    for name in required:
        tensor = by_name[name]
        probe = tensor.pytorch_probe
        source = name if probe is None or probe.source is None else probe.source
        matched = _copy_trace_artifacts(
            artifacts=artifacts,
            trace_artifacts=trace.tensors,
            source=source,
            tensor_name=name,
        )
        matched = (
            _copy_trace_artifacts(
                artifacts=artifacts,
                trace_artifacts=trace.tokens,
                source=source,
                tensor_name=name,
            )
            or matched
        )
        if not matched:
            raise ValueError(
                f"Reference trace missing artifact for {name!r} "
                f"(probe source {source!r})"
            )
    return artifacts


def _cached_trace_complete(
    cached: Mapping[str, torch.Tensor],
    *,
    required: tuple[str, ...],
) -> bool:
    for name in required:
        if name in cached:
            continue
        suffix = f".{name}"
        if any(key.endswith(suffix) for key in cached):
            continue
        return False
    return True


def _copy_trace_artifacts(
    *,
    artifacts: dict[str, torch.Tensor],
    trace_artifacts: Mapping[str, torch.Tensor],
    source: str,
    tensor_name: str,
) -> bool:
    matched = False
    for trace_key, value in trace_artifacts.items():
        artifact_key = _map_trace_key(trace_key, source=source, tensor_name=tensor_name)
        if artifact_key is None:
            continue
        artifacts[artifact_key] = value.detach().cpu().contiguous()
        matched = True
    return matched


def _map_trace_key(trace_key: str, *, source: str, tensor_name: str) -> str | None:
    if trace_key == source:
        return tensor_name
    suffix = f".{source}"
    if not trace_key.endswith(suffix):
        return None
    return f"{trace_key[: -len(suffix)]}.{tensor_name}"


def _trace_reference_fingerprint(
    *,
    provider_id: str,
    tensors: tuple[LogicalTensor, ...],
    inputs: Mapping[str, Any],
    required: tuple[str, ...],
    extra: Mapping[str, Any],
) -> str:
    payload = {
        "provider": provider_id,
        "required": required,
        "sources": {
            tensor.name: None if tensor.pytorch_probe is None else tensor.pytorch_probe.source
            for tensor in tensors
            if tensor.name in required
        },
        "inputs": {name: _jsonable(value) for name, value in sorted(inputs.items())},
        "extra": {name: _jsonable(value) for name, value in sorted(extra.items())},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        digest = hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return {
            "dtype": str(value.dtype),
            "shape": tuple(int(dim) for dim in value.shape),
            "sha256": digest,
        }
    if isinstance(value, Mapping):
        mapping = cast("Mapping[Any, Any]", value)
        return {str(key): _jsonable(item) for key, item in sorted(mapping.items())}
    if isinstance(value, (tuple, list)):
        sequence = cast("tuple[Any, ...] | list[Any]", value)
        return [_jsonable(item) for item in sequence]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
