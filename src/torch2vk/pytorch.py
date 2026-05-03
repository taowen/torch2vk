"""PyTorch reference capture for logical tensors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from .logical import LogicalTensor, PyTorchProbe

type TransformFn = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]


class ReferenceProvider(Protocol):
    def ensure(
        self,
        *,
        tensors: tuple[LogicalTensor, ...],
        inputs: Mapping[str, Any],
        cache: ArtifactCache,
        transforms: Mapping[str, TransformFn] | None = None,
        extra_fingerprint: Mapping[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class PyTorchModelReferenceProvider:
    model: Any

    def ensure(
        self,
        *,
        tensors: tuple[LogicalTensor, ...],
        inputs: Mapping[str, Any],
        cache: ArtifactCache,
        transforms: Mapping[str, TransformFn] | None = None,
        extra_fingerprint: Mapping[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        return ensure_pytorch_reference(
            model=self.model,
            tensors=tensors,
            inputs=inputs,
            cache=cache,
            transforms=transforms,
            extra_fingerprint=extra_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class ArtifactCache:
    root: Path

    def load(self, fingerprint: str) -> dict[str, torch.Tensor]:
        directory = self.root / fingerprint
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return {}
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if raw.get("fingerprint") != fingerprint:
            raise ValueError(f"{manifest_path} fingerprint mismatch")
        artifacts: dict[str, torch.Tensor] = {}
        for name in raw.get("artifacts", []):
            if not isinstance(name, str) or not name:
                raise ValueError(f"{manifest_path} contains invalid artifact key {name!r}")
            artifacts[name] = torch.load(_artifact_path(directory, name), map_location="cpu")
        return artifacts

    def store(self, fingerprint: str, artifacts: Mapping[str, torch.Tensor]) -> None:
        directory = self.root / fingerprint
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in artifacts.items():
            torch.save(value.detach().cpu().contiguous(), _artifact_path(directory, name))
        manifest = {
            "fingerprint": fingerprint,
            "artifacts": sorted(artifacts),
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class PyTorchForwardCapture:
    model: Any
    tensors: tuple[LogicalTensor, ...]
    inputs: Mapping[str, Any]
    transforms: Mapping[str, TransformFn] | None = None

    def run(self, names: Iterable[str]) -> dict[str, torch.Tensor]:
        required = set(names)
        artifacts: dict[str, torch.Tensor] = {}
        handles: list[Any] = []
        by_name = {tensor.name: tensor for tensor in self.tensors}
        try:
            for name in required:
                tensor = by_name[name]
                probe = tensor.pytorch_probe
                if probe is None:
                    raise ValueError(f"{name} has no PyTorch probe")
                if probe.kind == "module_output":
                    handles.append(_install_module_output_probe(self.model, tensor, artifacts))
                elif probe.kind == "module_input":
                    handles.append(_install_module_input_probe(self.model, tensor, artifacts))
                elif probe.kind in {"manual", "derived"}:
                    continue
                else:
                    raise ValueError(f"{name} has unsupported probe kind {probe.kind!r}")
            self.model.eval()
            with torch.no_grad():
                output = self.model(**self.inputs)
            _collect_manual_probes(output, by_name, required, artifacts)
            _run_derived_probes(by_name, required, artifacts, self.transforms or {})
        finally:
            for handle in handles:
                handle.remove()
        missing = required - artifacts.keys()
        if missing:
            raise ValueError(f"Missing PyTorch artifacts: {sorted(missing)}")
        return {name: artifacts[name] for name in required}


def ensure_pytorch_reference(
    *,
    model: Any,
    tensors: tuple[LogicalTensor, ...],
    inputs: Mapping[str, Any],
    cache: ArtifactCache,
    transforms: Mapping[str, TransformFn] | None = None,
    extra_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    required = required_probe_names(tensors)
    fingerprint = reference_fingerprint(
        tensors=tensors,
        inputs=inputs,
        required=required,
        extra=extra_fingerprint or {},
    )
    cached = cache.load(fingerprint)
    if _cached_reference_complete(cached, tensors=tensors, required=required):
        return {name: cached[name] for name in required}
    captured = PyTorchForwardCapture(
        model=model,
        tensors=tensors,
        inputs=inputs,
        transforms=transforms,
    ).run(required)
    cache.store(fingerprint, captured)
    return captured


def reference_fingerprint(
    *,
    tensors: tuple[LogicalTensor, ...],
    inputs: Mapping[str, Any],
    required: tuple[str, ...],
    extra: Mapping[str, Any],
) -> str:
    payload = {
        "required": required,
        "probes": {
            tensor.name: _tensor_payload(tensor)
            for tensor in tensors
            if tensor.name in required
        },
        "inputs": {name: _input_payload(value) for name, value in sorted(inputs.items())},
        "extra": dict(extra),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def required_probe_names(tensors: Iterable[LogicalTensor]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for tensor in tensors:
        if tensor.pytorch_probe is None or tensor.compare is None or tensor.name in seen:
            continue
        seen.add(tensor.name)
        names.append(tensor.name)
    return tuple(names)


def _cached_reference_complete(
    cached: Mapping[str, torch.Tensor],
    *,
    tensors: tuple[LogicalTensor, ...],
    required: tuple[str, ...],
) -> bool:
    by_name = {tensor.name: tensor for tensor in tensors}
    for name in required:
        artifact = cached.get(name)
        if artifact is None:
            return False
        _validate_cached_artifact(by_name[name], artifact)
    return True


def _validate_cached_artifact(tensor: LogicalTensor, artifact: torch.Tensor) -> None:
    expected_shape = tensor.shape
    if any(not isinstance(dim, int) for dim in expected_shape):
        raise ValueError(f"{tensor.name} has unresolved symbolic shape {expected_shape}")
    actual_shape = tuple(int(dim) for dim in artifact.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{tensor.name} cached PyTorch artifact shape mismatch: "
            f"expected {expected_shape}, got {actual_shape}"
        )
    expected_dtype = _normalized_torch_dtype(tensor)
    if expected_dtype is not None and artifact.dtype is not expected_dtype:
        raise ValueError(
            f"{tensor.name} cached PyTorch artifact dtype mismatch: "
            f"expected {expected_dtype}, got {artifact.dtype}"
        )
    if not artifact.is_contiguous():
        raise ValueError(f"{tensor.name} cached PyTorch artifact is not contiguous")


def _install_module_output_probe(
    model: Any,
    tensor: LogicalTensor,
    artifacts: dict[str, torch.Tensor],
) -> Any:
    probe = _require_probe(tensor)
    module = _resolve_module_path(model, _require_target(probe))

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        artifacts[tensor.name] = _normalize_probe_value(
            _select_probe_value(output, probe.selector),
            tensor,
            probe,
        )

    return module.register_forward_hook(hook)


def _install_module_input_probe(
    model: Any,
    tensor: LogicalTensor,
    artifacts: dict[str, torch.Tensor],
) -> Any:
    probe = _require_probe(tensor)
    module = _resolve_module_path(model, _require_target(probe))

    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        artifacts[tensor.name] = _normalize_probe_value(inputs[probe.index], tensor, probe)

    return module.register_forward_pre_hook(hook)


def _collect_manual_probes(
    output: Any,
    tensors: Mapping[str, LogicalTensor],
    required: set[str],
    artifacts: dict[str, torch.Tensor],
) -> None:
    for name in required:
        tensor = tensors[name]
        probe = tensor.pytorch_probe
        if probe is None or probe.kind != "manual":
            continue
        source = probe.source
        if source is None:
            raise ValueError(f"{name} manual probe has no source")
        value = _select_probe_value(output, source)
        artifacts[name] = _normalize_probe_value(value, tensor, probe)


def _run_derived_probes(
    tensors: Mapping[str, LogicalTensor],
    required: set[str],
    artifacts: dict[str, torch.Tensor],
    transforms: Mapping[str, TransformFn],
) -> None:
    pending = True
    while pending:
        pending = False
        for name in required:
            if name in artifacts:
                continue
            tensor = tensors[name]
            probe = tensor.pytorch_probe
            if probe is None or probe.kind != "derived":
                continue
            if not set(probe.inputs).issubset(artifacts):
                continue
            if probe.transform is None:
                raise ValueError(f"{name} derived probe has no transform")
            try:
                transform = transforms[probe.transform]
            except KeyError as exc:
                raise KeyError(f"Missing derived transform {probe.transform}") from exc
            artifacts[name] = _normalize_probe_value(transform(artifacts), tensor, probe)
            pending = True


def _resolve_module_path(model: Any, path: str) -> Any:
    current = model
    parts = path.split(".")
    if parts and parts[0] == "model" and not hasattr(current, "model"):
        parts = parts[1:]
    for part in parts:
        current = current[int(part)] if part.isdecimal() else getattr(current, part)
    return current


def _select_probe_value(value: Any, selector: str | None) -> Any:
    if selector is None:
        return value
    current = value
    for part in selector.split("."):
        current = current[int(part)] if part.isdecimal() else getattr(current, part)
    return current


def _normalize_probe_value(value: Any, tensor: LogicalTensor, probe: PyTorchProbe) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{tensor.name} probe returned {type(value).__name__}, expected Tensor")
    result = value.detach().cpu()
    if probe.normalize == "float32_contiguous" or tensor.dtype in {"float32", "f32"}:
        result = result.float()
    elif tensor.dtype in {"int32", "i32"}:
        result = result.to(torch.int32)
    elif tensor.dtype in {"int64", "i64"}:
        result = result.to(torch.int64)
    elif tensor.dtype in {"float16", "f16"}:
        result = result.to(torch.float16)
    elif tensor.dtype in {"bfloat16", "bf16"}:
        result = result.to(torch.bfloat16)
    return result.contiguous()


def _require_probe(tensor: LogicalTensor) -> PyTorchProbe:
    if tensor.pytorch_probe is None:
        raise ValueError(f"{tensor.name} has no PyTorch probe")
    return tensor.pytorch_probe


def _require_target(probe: PyTorchProbe) -> str:
    if probe.target is None:
        raise ValueError(f"{probe.kind} probe has no target")
    return probe.target


def _probe_payload(probe: PyTorchProbe | None) -> object:
    if probe is None:
        return None
    return {
        "kind": probe.kind,
        "target": probe.target,
        "index": probe.index,
        "source": probe.source,
        "inputs": probe.inputs,
        "transform": probe.transform,
        "selector": probe.selector,
        "normalize": probe.normalize,
    }


def _tensor_payload(tensor: LogicalTensor) -> object:
    return {
        "dtype": tensor.dtype,
        "shape": tensor.shape,
        "probe": _probe_payload(tensor.pytorch_probe),
        "compare": None
        if tensor.compare is None
        else {
            "kind": tensor.compare.kind,
            "rtol": tensor.compare.rtol,
            "atol": tensor.compare.atol,
            "max_abs": tensor.compare.max_abs,
        },
    }


def _normalized_torch_dtype(tensor: LogicalTensor) -> torch.dtype | None:
    probe = tensor.pytorch_probe
    if probe is not None and probe.normalize == "float32_contiguous":
        return torch.float32
    aliases = {
        "float32": torch.float32,
        "f32": torch.float32,
        "int32": torch.int32,
        "i32": torch.int32,
        "int64": torch.int64,
        "i64": torch.int64,
        "float16": torch.float16,
        "f16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return aliases.get(tensor.dtype)


def _input_payload(value: Any) -> object:
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        digest_tensor = cpu
        if digest_tensor.dtype is torch.bfloat16:
            digest_tensor = digest_tensor.view(torch.uint16)
        digest = hashlib.sha256(digest_tensor.numpy().tobytes()).hexdigest()
        return {
            "dtype": str(cpu.dtype),
            "shape": tuple(int(dim) for dim in cpu.shape),
            "sha256": digest,
        }
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    if isinstance(value, tuple | list):
        return [_input_payload(item) for item in cast("list[Any] | tuple[Any, ...]", value)]
    return repr(value)


def _artifact_path(directory: Path, name: str) -> Path:
    return directory / f"{name.replace('/', '_').replace('.', '__')}.pt"
