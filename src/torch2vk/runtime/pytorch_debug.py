"""PyTorch/reference compare and drilldown support for RuntimeSession."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeGuard

import numpy as np

from torch2vk.runtime.compare import (
    CompareAssertionError,
    TensorCompareResult,
    as_numpy_array,
    compare_arrays,
    select_probe_value,
    write_compare_summary,
)
from torch2vk.runtime.frame import FrameContext
from torch2vk.runtime.logical import LogicalTensor, TensorRole
from torch2vk.runtime.shader import DispatchRecord

if TYPE_CHECKING:
    import torch

    from torch2vk.runtime.session import RuntimeSession


_PYTORCH_ARTIFACT_CACHE_VERSION = 2


class _PyTorchModuleLike(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _NamedModulesLike(Protocol):
    def named_modules(self) -> Iterable[tuple[str, object]]: ...


class _HookHandleLike(Protocol):
    def remove(self) -> None: ...


class _ForwardHookModuleLike(Protocol):
    def register_forward_hook(self, hook: object) -> _HookHandleLike: ...


class _TorchTensorLike(Protocol):
    def detach(self) -> "_TorchTensorLike": ...

    def clone(self) -> "_TorchTensorLike": ...

    def cpu(self) -> "_TorchTensorLike": ...

    def float(self) -> "_TorchTensorLike": ...

    def numpy(self) -> np.ndarray: ...


class _PyTorchModelLoader(Protocol):
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        *,
        dtype: object,
        attn_implementation: str,
    ) -> "_LoadedPyTorchModelLike": ...


class _LoadedPyTorchModelLike(_PyTorchModuleLike, Protocol):
    def to(self, device: object) -> object: ...

    def eval(self) -> object: ...


class _DeviceCarrier(Protocol):
    @property
    def device(self) -> "torch.device": ...


class _ParametersLike(Protocol):
    def parameters(self) -> Iterable[object]: ...


@dataclass(frozen=True, slots=True)
class _PyTorchCachePaths:
    root: Path
    array: Path
    metadata: Path
    metadata_payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class _ReadCompareOutcome:
    failed_result: TensorCompareResult | None
    missing_reference_tensors: tuple[str, ...]
    unprobed_read_tensors: tuple[str, ...]
    ancestor_probe_tensors: tuple[str, ...]


class _PyTorchDebugRunner:
    def __init__(self, rt: RuntimeSession) -> None:
        self.rt = rt

    def __getattr__(self, name: str) -> object:
        return getattr(self.rt, name)

    def _compare_frame(self, frame: FrameContext) -> None:
        written = _unique_tensors(frame.written_tensors)
        if frame.pytorch_model is None:
            return

        probe_targets = _default_frame_compare_targets(
            tensor for tensor in written if tensor.pytorch_probe is not None
        )
        if not probe_targets:
            if _is_stateful_pytorch_frame(frame):
                self._run_pytorch_probes(frame, [])
            return
        expected_by_tensor = self._expected_pytorch_artifacts(frame, probe_targets)
        for tensor in probe_targets:
            try:
                expected = expected_by_tensor[tensor.name]
            except KeyError as exc:
                raise KeyError(f"PyTorch probe did not capture artifact for {tensor.name}") from exc
            self._record_compare(
                tensor=tensor,
                frame=frame,
                candidate=self.readback(tensor),
                expected=expected,
            )

    def _expected_pytorch_artifacts(
        self,
        frame: FrameContext,
        tensors: list[LogicalTensor],
    ) -> dict[str, object]:
        if frame.pytorch_model is None:
            return {}
        probe_targets = [tensor for tensor in tensors if tensor.pytorch_probe is not None]
        if not probe_targets:
            return {}
        expected_by_tensor = {
            tensor.name: frame.pytorch_captured_artifacts[tensor.name]
            for tensor in probe_targets
            if tensor.name in frame.pytorch_captured_artifacts
        }
        missing_probe_targets = [
            tensor for tensor in probe_targets if tensor.name not in expected_by_tensor
        ]
        if missing_probe_targets and not _is_stateful_pytorch_frame(frame):
            cached, missing_probe_targets = self._load_cached_pytorch_artifacts(
                frame,
                missing_probe_targets,
                frame.pytorch_model,
            )
            expected_by_tensor.update(cached)
        if missing_probe_targets:
            captured = self._run_pytorch_probes(frame, missing_probe_targets)
            expected_by_tensor.update(captured)
            self._store_cached_pytorch_artifacts(
                frame,
                missing_probe_targets,
                frame.pytorch_model,
                captured,
            )
        return expected_by_tensor

    def _record_compare(
        self,
        *,
        tensor: LogicalTensor,
        frame: FrameContext,
        candidate: object,
        expected: object,
    ) -> None:
        try:
            result = compare_arrays(
                tensor=tensor,
                frame=frame.frame,
                candidate=candidate,
                expected=expected,
                artifact_dir=self.artifact_dir,
                nearest_upstream_artifact_key=self._nearest_passed_artifact_key(frame.frame),
            )
        except CompareAssertionError as exc:
            result = self._attach_writer_drilldown(exc.result, frame=frame, seen=set())
            write_compare_summary(result)
            self._compare_results.append(result)
            raise CompareAssertionError(
                policy=result.tensor.compare or exc.policy,
                result=result,
            ) from exc
        self._compare_results.append(result)

    def _nearest_passed_artifact_key(self, frame: str) -> str | None:
        for result in reversed(self._compare_results):
            if result.frame == frame and result.passed:
                return result.artifact_key
        return None

    def _attach_writer_drilldown(
        self,
        result: TensorCompareResult,
        *,
        frame: FrameContext,
        seen: set[str],
    ) -> TensorCompareResult:
        initial_artifact_key = result.artifact_key
        current = result
        path: list[dict[str, object]] = []

        while True:
            if current.artifact_key in seen:
                return self._with_drilldown_classification(
                    current,
                    classification="cycle_detected",
                    report_path=None,
                    input_paths=(),
                    output_paths=(),
                )
            seen.add(current.artifact_key)

            writer = current.tensor.writer
            if writer is None:
                return self._with_drilldown_classification(
                    current,
                    classification="writer_not_recorded",
                    report_path=None,
                    input_paths=(),
                    output_paths=(),
                )
            try:
                record = self._dispatch_records[writer.dispatch_index]
            except IndexError:
                return self._with_drilldown_classification(
                    current,
                    classification="writer_record_missing",
                    report_path=None,
                    input_paths=(),
                    output_paths=(),
                )
            if record.frame != current.frame:
                return self._with_drilldown_classification(
                    current,
                    classification="writer_frame_mismatch",
                    report_path=None,
                    input_paths=(),
                    output_paths=(),
                )

            root = (
                self.artifact_dir
                / "debug"
                / _safe_path_component(current.frame)
                / _safe_path_component(current.tensor.name)
                / "writer_io"
            )
            root.mkdir(parents=True, exist_ok=True)
            input_paths: list[str] = []
            output_paths: list[str] = []
            reads: list[dict[str, object]] = []
            writes: list[dict[str, object]] = []

            for field_name, tensor in record.reads:
                entry = self._dump_debug_tensor(
                    root=root, prefix="read", field_name=field_name, tensor=tensor
                )
                reads.append(entry)
                path_value = entry.get("path")
                if isinstance(path_value, str):
                    input_paths.append(path_value)
            for field_name, tensor in record.writes:
                entry = self._dump_debug_tensor(
                    root=root, prefix="write", field_name=field_name, tensor=tensor
                )
                writes.append(entry)
                path_value = entry.get("path")
                if isinstance(path_value, str):
                    output_paths.append(path_value)

            outcome = self._compare_declared_read_tensors(frame=frame, record=record, seen=seen)
            if outcome.failed_result is not None:
                classification = "upstream_input_bad"
            elif outcome.missing_reference_tensors:
                classification = "missing_reference_probe"
            elif outcome.unprobed_read_tensors:
                classification = "unprobed_input_gap"
            else:
                classification = "input_ok_output_bad"

            dispatch = _serializable_dispatch_record(record)
            dispatch_path = root / "dispatch.json"
            dispatch_path.write_text(
                json.dumps(dispatch, indent=2, sort_keys=True), encoding="utf-8"
            )
            read_status = [
                self._upstream_tensor_status(frame=current.frame, tensor=tensor)
                for _, tensor in record.reads
            ]
            path.append(
                {
                    "failed_artifact": current.artifact_key,
                    "tensor": current.tensor.name,
                    "writer": {
                        "frame": writer.frame,
                        "shader": writer.shader,
                        "dispatch_index": writer.dispatch_index,
                    },
                    "classification": classification,
                    "missing_reference_tensors": outcome.missing_reference_tensors,
                    "unprobed_read_tensors": outcome.unprobed_read_tensors,
                    "ancestor_probe_tensors": outcome.ancestor_probe_tensors,
                }
            )
            report = {
                "classification": classification,
                "initial_failed_artifact": initial_artifact_key,
                "failed_artifact": current.artifact_key,
                "nearest_upstream_artifact_key": current.nearest_upstream_artifact_key,
                "missing_reference_tensors": outcome.missing_reference_tensors,
                "unprobed_read_tensors": outcome.unprobed_read_tensors,
                "ancestor_probe_tensors": outcome.ancestor_probe_tensors,
                "dispatch": dispatch,
                "dispatch_artifact_path": str(dispatch_path),
                "reads": reads,
                "writes": writes,
                "read_status": read_status,
                "drilldown_path": path,
            }
            report_path = root / "drilldown.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            current = self._with_drilldown_classification(
                current,
                classification=classification,
                report_path=report_path,
                input_paths=tuple(input_paths),
                output_paths=tuple(output_paths),
            )
            if outcome.failed_result is None:
                return current
            current = outcome.failed_result

    def _compare_declared_read_tensors(
        self,
        *,
        frame: FrameContext,
        record: DispatchRecord,
        seen: set[str],
    ) -> _ReadCompareOutcome:
        read_tensors = _unique_tensors(tensor for _, tensor in record.reads)
        probe_targets = [
            tensor
            for tensor in read_tensors
            if tensor.compare is not None
            and tensor.pytorch_probe is not None
            and (
                f"{self._compare_frame_for_tensor(default_frame=frame, tensor=tensor).frame}"
                f"/{tensor.name}"
            )
            not in seen
        ]
        non_probed_reads = [
            tensor
            for tensor in read_tensors
            if tensor.role is not TensorRole.WEIGHT
            and not (tensor.role is TensorRole.INPUT and tensor.writer is None)
            and (tensor.compare is None or tensor.pytorch_probe is None)
        ]
        unprobed_read_tensors = tuple(tensor.name for tensor in non_probed_reads)
        still_missing: list[str] = []
        ancestor_probe_tensors: list[str] = []
        for tensor in non_probed_reads:
            ancestors = self._search_probed_ancestors(tensor, frame, seen)
            if ancestors:
                probe_targets.extend(ancestors)
                ancestor_probe_tensors.extend(ancestor.name for ancestor in ancestors)
            else:
                still_missing.append(tensor.name)
        missing_reference_tensors = tuple(still_missing)
        probe_targets = _unique_tensors(probe_targets)
        ancestor_probe_tensor_names = tuple(dict.fromkeys(ancestor_probe_tensors))
        if not probe_targets:
            return _ReadCompareOutcome(
                failed_result=None,
                missing_reference_tensors=missing_reference_tensors,
                unprobed_read_tensors=unprobed_read_tensors,
                ancestor_probe_tensors=ancestor_probe_tensor_names,
            )
        expected_by_frame: dict[str, dict[str, object]] = {}
        for tensor in probe_targets:
            probe_frame = self._compare_frame_for_tensor(default_frame=frame, tensor=tensor)
            expected_by_tensor = expected_by_frame.get(probe_frame.frame)
            if expected_by_tensor is None:
                frame_targets = [
                    candidate
                    for candidate in probe_targets
                    if self._compare_frame_for_tensor(default_frame=frame, tensor=candidate).frame
                    == probe_frame.frame
                ]
                expected_by_tensor = self._expected_pytorch_artifacts(probe_frame, frame_targets)
                expected_by_frame[probe_frame.frame] = expected_by_tensor
            expected = expected_by_tensor.get(tensor.name)
            if expected is None:
                missing_reference_tensors = (*missing_reference_tensors, tensor.name)
                continue
            try:
                result = compare_arrays(
                    tensor=tensor,
                    frame=probe_frame.frame,
                    candidate=self.readback(tensor),
                    expected=expected,
                    artifact_dir=self.artifact_dir,
                    nearest_upstream_artifact_key=self._nearest_passed_artifact_key(
                        probe_frame.frame
                    ),
                )
            except CompareAssertionError as exc:
                write_compare_summary(exc.result)
                self._compare_results.append(exc.result)
                return _ReadCompareOutcome(
                    failed_result=exc.result,
                    missing_reference_tensors=missing_reference_tensors,
                    unprobed_read_tensors=unprobed_read_tensors,
                    ancestor_probe_tensors=ancestor_probe_tensor_names,
                )
            self._compare_results.append(result)
        return _ReadCompareOutcome(
            failed_result=None,
            missing_reference_tensors=missing_reference_tensors,
            unprobed_read_tensors=unprobed_read_tensors,
            ancestor_probe_tensors=ancestor_probe_tensor_names,
        )

    def _search_probed_ancestors(
        self,
        tensor: LogicalTensor,
        frame: FrameContext,
        seen: set[str],
        *,
        max_depth: int = 100,
    ) -> list[LogicalTensor]:
        """BFS through writer graph to find nearest tensors with compare + probe."""
        queue = [tensor]
        visited: set[str] = set()
        found: list[LogicalTensor] = []
        depth = 0
        while queue and depth < max_depth:
            next_queue: list[LogicalTensor] = []
            for t in queue:
                if t.name in visited:
                    continue
                visited.add(t.name)
                writer = t.writer
                if writer is None:
                    continue
                try:
                    record = self._dispatch_records[writer.dispatch_index]
                except IndexError:
                    continue
                for _, read_tensor in record.reads:
                    if read_tensor.role is TensorRole.WEIGHT:
                        continue
                    if read_tensor.name in visited:
                        continue
                    key = (
                        f"{self._compare_frame_for_tensor(default_frame=frame, tensor=read_tensor).frame}"
                        f"/{read_tensor.name}"
                    )
                    if key in seen:
                        continue
                    if read_tensor.compare is not None and read_tensor.pytorch_probe is not None:
                        found.append(read_tensor)
                    else:
                        next_queue.append(read_tensor)
            queue = next_queue
            depth += 1
        return found

    def _compare_frame_for_tensor(
        self, *, default_frame: FrameContext, tensor: LogicalTensor
    ) -> FrameContext:
        writer = tensor.writer
        if writer is None or writer.frame == default_frame.frame:
            return default_frame
        return self._frame_history.get(writer.frame, default_frame)

    def _with_drilldown_classification(
        self,
        result: TensorCompareResult,
        *,
        classification: str,
        report_path: Path | None,
        input_paths: tuple[str, ...],
        output_paths: tuple[str, ...],
    ) -> TensorCompareResult:
        return replace(
            result,
            drilldown_classification=classification,
            drilldown_artifact_path=None if report_path is None else str(report_path),
            writer_input_artifact_paths=input_paths,
            writer_output_artifact_paths=output_paths,
        )

    def _upstream_tensor_status(self, *, frame: str, tensor: LogicalTensor) -> dict[str, object]:
        artifact_key = f"{frame}/{tensor.name}"
        latest_result = None
        for result in reversed(self._compare_results):
            if result.artifact_key == artifact_key:
                latest_result = result
                break
        writer = tensor.writer
        return {
            "artifact_key": artifact_key,
            "tensor": tensor.name,
            "has_compare_policy": tensor.compare is not None,
            "has_pytorch_probe": tensor.pytorch_probe is not None,
            "latest_compare": None
            if latest_result is None
            else {
                "passed": latest_result.passed,
                "failure_reason": latest_result.failure_reason,
                "max_abs": latest_result.max_abs,
                "max_rel": latest_result.max_rel,
            },
            "writer": None
            if writer is None
            else {
                "frame": writer.frame,
                "shader": writer.shader,
                "dispatch_index": writer.dispatch_index,
            },
        }

    def _dump_debug_tensor(
        self,
        *,
        root: Path,
        prefix: str,
        field_name: str,
        tensor: LogicalTensor,
    ) -> dict[str, object]:
        entry: dict[str, object] = {
            "field": field_name,
            "tensor": tensor.name,
            "shape": tensor.concrete_shape,
            "dtype": tensor.spec.dtype,
            "layout": repr(tensor.layout),
        }
        if tensor.buffer is None:
            entry["materialized"] = False
            return entry
        array = self.readback(tensor)
        path = (
            root
            / f"{prefix}_{_safe_path_component(field_name)}_{_safe_path_component(tensor.name)}.npy"
        )
        np.save(path, array)
        entry.update(
            {
                "materialized": True,
                "path": str(path),
                "min": _finite_stat(array, "min"),
                "max": _finite_stat(array, "max"),
                "mean": _finite_stat(array, "mean"),
                "nan_count": int(np.isnan(array.astype(np.float64, copy=False)).sum()),
                "inf_count": int(np.isinf(array.astype(np.float64, copy=False)).sum()),
            }
        )
        return entry

    def _load_cached_pytorch_artifacts(
        self,
        frame: FrameContext,
        tensors: list[LogicalTensor],
        model: object,
    ) -> tuple[dict[str, object], list[LogicalTensor]]:
        cached: dict[str, object] = {}
        missing: list[LogicalTensor] = []
        for tensor in tensors:
            paths = self._pytorch_cache_paths(frame=frame, tensor=tensor, model=model)
            if not paths.array.is_file() or not paths.metadata.is_file():
                missing.append(tensor)
                continue
            try:
                metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                missing.append(tensor)
                continue
            if metadata != _json_normalized(paths.metadata_payload):
                missing.append(tensor)
                continue
            cached[tensor.name] = np.load(paths.array, allow_pickle=False)
        return cached, missing

    def _store_cached_pytorch_artifacts(
        self,
        frame: FrameContext,
        tensors: list[LogicalTensor],
        model: object,
        captured: Mapping[str, object],
    ) -> None:
        for tensor in tensors:
            if tensor.name not in captured:
                continue
            paths = self._pytorch_cache_paths(frame=frame, tensor=tensor, model=model)
            paths.root.mkdir(parents=True, exist_ok=True)
            array = as_numpy_array(captured[tensor.name])
            np.save(paths.array, array)
            paths.metadata.write_text(
                json.dumps(_json_normalized(paths.metadata_payload), indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def _pytorch_cache_paths(
        self,
        *,
        frame: FrameContext,
        tensor: LogicalTensor,
        model: object,
    ) -> _PyTorchCachePaths:
        metadata = self._pytorch_cache_metadata(frame=frame, tensor=tensor, model=model)
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        root = (
            self.artifact_dir
            / "pytorch_cache"
            / _safe_path_component(frame.frame)
            / _safe_path_component(tensor.name)
            / digest
        )
        return _PyTorchCachePaths(
            root=root,
            array=root / "expected.npy",
            metadata=root / "metadata.json",
            metadata_payload=metadata,
        )

    def _pytorch_cache_metadata(
        self,
        *,
        frame: FrameContext,
        tensor: LogicalTensor,
        model: object,
    ) -> dict[str, object]:
        probe = tensor.pytorch_probe
        return {
            "version": _PYTORCH_ARTIFACT_CACHE_VERSION,
            "frame": frame.frame,
            "artifact_key": f"{frame.frame}/{tensor.name}",
            "tensor": {
                "name": tensor.name,
                "shape": tensor.concrete_shape,
                "dtype": tensor.spec.dtype,
                "layout": repr(tensor.layout),
                "compare": None if tensor.compare is None else asdict(tensor.compare),
            },
            "probe": None if probe is None else asdict(probe),
            "inputs": self._frame_input_fingerprints(frame),
            "pytorch_frame": {
                "input_prefixes": _pytorch_input_prefixes(frame),
                "cache_policy": frame.pytorch_cache_policy,
                "cache_namespace": frame.pytorch_cache_namespace,
                "reset_cache": frame.pytorch_reset_cache,
            },
            "explicit_pytorch_args": [
                self._pytorch_value_fingerprint(value) for value in frame.pytorch_args
            ],
            "explicit_pytorch_kwargs": {
                key: self._pytorch_value_fingerprint(value)
                for key, value in sorted(frame.pytorch_kwargs.items())
            },
            "model": self._model_fingerprint(model),
        }

    def _frame_input_fingerprints(self, frame: FrameContext) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for tensor, value in self._inputs.items():
            if tensor.role is not TensorRole.INPUT:
                continue
            if not any(tensor.name.startswith(prefix) for prefix in _pytorch_input_prefixes(frame)):
                continue
            entries.append(
                {
                    "tensor": tensor.name,
                    "dtype": tensor.spec.dtype,
                    "shape": tensor.concrete_shape,
                    "value": _value_fingerprint(value),
                }
            )
        return sorted(entries, key=lambda entry: str(entry["tensor"]))

    def _model_fingerprint(self, model: object) -> dict[str, object]:
        model_type = type(model)
        fingerprint: dict[str, object] = {
            "type": f"{model_type.__module__}.{model_type.__qualname__}",
        }
        if self.model_dir is not None:
            files = [
                path
                for pattern in ("*.safetensors", "*.safetensors.index.json", "config.json")
                for path in sorted(self.model_dir.glob(pattern))
                if path.is_file()
            ]
            fingerprint.update(
                {
                    "kind": "model_dir",
                    "model_dir": str(self.model_dir),
                    "files": [
                        {
                            "path": str(path.relative_to(self.model_dir)),
                            "size": path.stat().st_size,
                            "mtime_ns": path.stat().st_mtime_ns,
                        }
                        for path in files
                    ],
                }
            )
            return fingerprint
        fingerprint.update({"kind": "process_object", "object_id": id(model)})
        return fingerprint

    def _pytorch_value_fingerprint(self, value: object) -> dict[str, object]:
        if isinstance(value, LogicalTensor):
            try:
                registered = self._inputs[value]
            except KeyError:
                return {"kind": "logical_tensor", "tensor": value.name, "registered": False}
            return {
                "kind": "logical_tensor",
                "tensor": value.name,
                "registered": True,
                "value": _value_fingerprint(registered),
            }
        return _value_fingerprint(value)

    def _run_pytorch_probes(
        self,
        frame: FrameContext,
        tensors: list[LogicalTensor],
    ) -> dict[str, object]:
        import torch

        model = frame.pytorch_model
        if model is None:
            raise RuntimeError("PyTorch probe requested without pytorch_model")
        if not _is_pytorch_module_like(model):
            raise TypeError(f"pytorch_model must be callable, got {type(model).__name__}")
        module_like = model
        modules = dict(model.named_modules()) if _has_named_modules(model) else {}
        captured: dict[str, object] = {}
        hooks: list[_HookHandleLike] = []
        root_output_tensors: list[LogicalTensor] = []

        for tensor in tensors:
            probe = tensor.pytorch_probe
            if probe is None:
                continue
            if probe.kind == "derived":
                raise NotImplementedError(f"{tensor.name} derived PyTorchProbe is not implemented")
            if probe.target == "":
                root_output_tensors.append(tensor)
                continue
            try:
                module = modules[probe.target]
            except KeyError as exc:
                raise KeyError(f"PyTorch module probe target not found: {probe.target}") from exc
            if not _is_forward_hook_module(module):
                raise TypeError(
                    f"PyTorch probe target {probe.target!r} does not support forward hooks"
                )
            if probe.kind == "module_output":
                hooks.append(
                    module.register_forward_hook(
                        _make_output_hook(tensor=tensor, captured=captured)
                    )
                )
            elif probe.kind == "module_input":
                hooks.append(
                    module.register_forward_hook(_make_input_hook(tensor=tensor, captured=captured))
                )
            else:
                raise NotImplementedError(
                    f"{tensor.name} unsupported PyTorchProbe kind: {probe.kind}"
                )

        cache_overrides, update_cache_state = self._prepare_pytorch_cache_probe_run(
            frame=frame,
            model=model,
        )
        args, kwargs = self._pytorch_forward_inputs(
            frame,
            model,
            cache_overrides=cache_overrides,
        )
        try:
            with torch.no_grad():
                output = module_like(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        frame.pytorch_forward_ran = True
        self._capture_pytorch_cache_state(
            frame=frame,
            model=model,
            output=output,
            update_cache_state=update_cache_state,
        )

        for tensor in root_output_tensors:
            probe = tensor.pytorch_probe
            assert probe is not None
            if probe.kind != "module_output":
                raise NotImplementedError(f"{tensor.name} root probe only supports module_output")
            captured[tensor.name] = select_probe_value(
                output,
                index=probe.index,
                selector=probe.selector,
            )
        frame.pytorch_captured_artifacts.update(captured)
        return captured

    def _prepare_pytorch_cache_probe_run(
        self,
        *,
        frame: FrameContext,
        model: object,
    ) -> tuple[dict[str, object], bool]:
        if not _is_stateful_pytorch_frame(frame):
            return {}, True
        namespace = _pytorch_cache_namespace(frame=frame, model=model)
        if not frame.pytorch_forward_ran:
            cache = self._ensure_pytorch_cache_state(frame=frame, model=model)
            if cache is not None:
                frame.pytorch_cache_input_snapshots[namespace] = _clone_pytorch_cache(
                    cache,
                    model=model,
                )
            return {}, True
        try:
            snapshot = frame.pytorch_cache_input_snapshots[namespace]
        except KeyError:
            return {}, False
        return {namespace: _clone_pytorch_cache(snapshot, model=model)}, False

    def _capture_pytorch_cache_state(
        self,
        *,
        frame: FrameContext,
        model: object,
        output: object,
        update_cache_state: bool,
    ) -> None:
        if not update_cache_state or not _is_stateful_pytorch_frame(frame):
            return
        past_key_values = getattr(output, "past_key_values", None)
        if past_key_values is not None:
            namespace = _pytorch_cache_namespace(frame=frame, model=model)
            self._pytorch_cache_states[namespace] = past_key_values

    def _pytorch_forward_inputs(
        self,
        frame: FrameContext,
        model: object,
        *,
        cache_overrides: Mapping[str, object] | None = None,
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        parameter_names = _pytorch_forward_parameter_names(model)
        if frame.pytorch_args or frame.pytorch_kwargs:
            device = self._pytorch_model_device(model)
            kwargs = {
                key: self._to_device(self._resolve_pytorch_value(value), device)
                for key, value in frame.pytorch_kwargs.items()
            }
            self._configure_pytorch_cache_kwargs(
                frame=frame,
                model=model,
                parameter_names=parameter_names,
                kwargs=kwargs,
                cache_overrides={} if cache_overrides is None else cache_overrides,
            )
            return tuple(self._to_device(self._resolve_pytorch_value(value), device) for value in frame.pytorch_args), kwargs
        kwargs = self._infer_pytorch_kwargs(
            frame=frame,
            model=model,
            parameter_names=parameter_names,
            cache_overrides={} if cache_overrides is None else cache_overrides,
        )
        return (), kwargs

    def _infer_pytorch_kwargs(
        self,
        *,
        frame: FrameContext,
        model: object,
        parameter_names: set[str],
        cache_overrides: Mapping[str, object],
    ) -> dict[str, object]:
        import torch

        device = self._pytorch_model_device(model)
        kwargs: dict[str, object] = {}
        for tensor, value in self._inputs.items():
            if tensor.role is not TensorRole.INPUT:
                continue
            kwarg = None
            for prefix in _pytorch_input_prefixes(frame):
                if tensor.name.startswith(prefix):
                    kwarg = tensor.name.removeprefix(prefix)
                    break
            if kwarg is None:
                continue
            if kwarg in parameter_names:
                pytorch_value = self._as_pytorch_input(value)
                if isinstance(pytorch_value, torch.Tensor):
                    pytorch_value = pytorch_value.to(device)
                kwargs[kwarg] = pytorch_value
        self._configure_pytorch_cache_kwargs(
            frame=frame,
            model=model,
            parameter_names=parameter_names,
            kwargs=kwargs,
            cache_overrides=cache_overrides,
        )
        return kwargs

    def _configure_pytorch_cache_kwargs(
        self,
        *,
        frame: FrameContext,
        model: object,
        parameter_names: set[str],
        kwargs: dict[str, object],
        cache_overrides: Mapping[str, object],
    ) -> None:
        if not _is_stateful_pytorch_frame(frame):
            return
        if "use_cache" in parameter_names:
            kwargs.setdefault("use_cache", True)
        if "past_key_values" not in parameter_names:
            return
        namespace = _pytorch_cache_namespace(frame=frame, model=model)
        past_key_values = cache_overrides.get(namespace)
        if past_key_values is None:
            past_key_values = self._ensure_pytorch_cache_state(frame=frame, model=model)
        if past_key_values is not None:
            kwargs.setdefault("past_key_values", past_key_values)

    def _ensure_pytorch_cache_state(self, *, frame: FrameContext, model: object) -> object | None:
        namespace = _pytorch_cache_namespace(frame=frame, model=model)
        if frame.pytorch_reset_cache and not frame.pytorch_forward_ran:
            cache = _new_hf_dynamic_cache(model)
            if cache is not None:
                self._pytorch_cache_states[namespace] = cache
            return cache
        try:
            return self._pytorch_cache_states[namespace]
        except KeyError:
            cache = _new_hf_dynamic_cache(model)
            if cache is not None:
                self._pytorch_cache_states[namespace] = cache
            return cache

    def _resolve_pytorch_value(self, value: object) -> object:
        if isinstance(value, LogicalTensor):
            try:
                registered = self._inputs[value]
            except KeyError as exc:
                raise RuntimeError(
                    f"{value.name} is used as a PyTorch input but was not registered"
                ) from exc
            return self._as_pytorch_input(registered)
        if isinstance(value, tuple):
            return tuple(self._resolve_pytorch_value(v) for v in value)
        if isinstance(value, list):
            return [self._resolve_pytorch_value(v) for v in value]
        return self._as_pytorch_input(value)

    def _as_pytorch_input(self, value: object) -> object:
        import torch

        if _is_torch_tensor(value):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(value))
        return value

    def _to_device(self, value: object, device: "torch.device") -> object:
        import torch

        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(self._to_device(v, device) for v in value)
        if isinstance(value, list):
            return [self._to_device(v, device) for v in value]
        return value

    def _load_pytorch_model(self, model_class: object) -> object | None:
        if model_class in self._pytorch_models:
            return self._pytorch_models[model_class]
        if self.model_dir is None:
            return None
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not _is_pytorch_model_loader(model_class):
            raise TypeError(
                f"pytorch_model_class must provide from_pretrained, got {type(model_class).__name__}"
            )
        model = model_class.from_pretrained(
            str(self.model_dir),
            dtype=torch.float32,
            attn_implementation="eager",
        )
        model.to(device)
        model.eval()
        self._pytorch_models[model_class] = model
        return model

    def _pytorch_model_device(self, model: object) -> "torch.device":
        import torch

        if _has_parameters(model):
            for parameter in model.parameters():
                if _has_torch_device(parameter):
                    return parameter.device
        return torch.device("cpu")


def compare_frame(rt: RuntimeSession, frame: FrameContext) -> None:
    _PyTorchDebugRunner(rt)._compare_frame(frame)


def load_pytorch_model(rt: RuntimeSession, model_class: object) -> object | None:
    return _PyTorchDebugRunner(rt)._load_pytorch_model(model_class)


def _pytorch_forward_parameter_names(model: object) -> set[str]:
    forward = getattr(model, "forward", model)
    if not callable(forward):
        raise TypeError(f"PyTorch model forward is not callable: {type(model).__name__}")
    signature = inspect.signature(forward)
    return {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }


def _unique_tensors(tensors: Iterable[LogicalTensor]) -> list[LogicalTensor]:
    seen: set[int] = set()
    unique: list[LogicalTensor] = []
    for tensor in tensors:
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(tensor)
    return unique


def _default_frame_compare_targets(written: Iterable[LogicalTensor]) -> list[LogicalTensor]:
    comparable = [tensor for tensor in written if tensor.compare is not None]
    if not comparable:
        return []
    return [comparable[-1]]


def _pytorch_input_prefixes(frame: FrameContext) -> tuple[str, ...]:
    prefixes = [_normalize_pytorch_input_prefix(frame.frame)]
    prefixes.extend(
        _normalize_pytorch_input_prefix(prefix) for prefix in frame.pytorch_input_prefixes
    )
    return tuple(dict.fromkeys(prefixes))


def _normalize_pytorch_input_prefix(prefix: str) -> str:
    if not prefix:
        raise ValueError("PyTorch input prefix must be non-empty")
    return prefix if prefix.endswith(".") else f"{prefix}."


def _is_stateful_pytorch_frame(frame: FrameContext) -> bool:
    return frame.pytorch_cache_policy != "none"


def _pytorch_cache_namespace(*, frame: FrameContext, model: object) -> str:
    if frame.pytorch_cache_namespace is not None:
        return frame.pytorch_cache_namespace
    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__qualname__}:{id(model)}"


def _new_hf_dynamic_cache(model: object) -> object | None:
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return None
    text_model = getattr(model, "model", None)
    config = getattr(text_model, "config", getattr(model, "config", None))
    if config is None:
        return None
    return DynamicCache(config=config)


def _clone_pytorch_cache(cache: object, *, model: object) -> object:
    cloned = _clone_hf_dynamic_cache(cache=cache, model=model)
    if cloned is not None:
        return cloned
    try:
        return copy.deepcopy(cache)
    except Exception as exc:
        raise RuntimeError(
            "Stateful PyTorch debug frames need a cloneable cache so drilldown can "
            "rerun probes without advancing the live reference state"
        ) from exc


def _clone_hf_dynamic_cache(*, cache: object, model: object) -> object | None:
    if type(cache).__name__ != "DynamicCache" or not hasattr(cache, "layers"):
        return None
    cloned = _new_hf_dynamic_cache(model)
    if cloned is None or not hasattr(cloned, "layers"):
        return None
    layers = getattr(cache, "layers")
    cloned_layers = getattr(cloned, "layers")
    if not isinstance(layers, list) or not isinstance(cloned_layers, list):
        return None
    for layer_index, layer in enumerate(layers):
        if layer_index >= len(cloned_layers):
            return None
        if not bool(getattr(layer, "is_initialized", False)):
            continue
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not _is_torch_tensor(keys) or not _is_torch_tensor(values):
            return None
        cloned_layers[layer_index].update(
            keys.detach().clone(),
            values.detach().clone(),
        )
    return cloned


def _is_torch_tensor(value: object) -> TypeGuard[_TorchTensorLike]:
    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy")


def _is_pytorch_module_like(value: object) -> TypeGuard[_PyTorchModuleLike]:
    return callable(value)


def _has_named_modules(value: object) -> TypeGuard[_NamedModulesLike]:
    named_modules = getattr(value, "named_modules", None)
    return callable(named_modules)


def _is_forward_hook_module(value: object) -> TypeGuard[_ForwardHookModuleLike]:
    register_forward_hook = getattr(value, "register_forward_hook", None)
    return callable(register_forward_hook)


def _is_pytorch_model_loader(value: object) -> TypeGuard[_PyTorchModelLoader]:
    from_pretrained = getattr(value, "from_pretrained", None)
    return callable(from_pretrained)


def _has_parameters(value: object) -> TypeGuard[_ParametersLike]:
    parameters = getattr(value, "parameters", None)
    return callable(parameters)


def _has_torch_device(value: object) -> TypeGuard[_DeviceCarrier]:
    return hasattr(value, "device")


def _value_fingerprint(value: object) -> dict[str, object]:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "numpy",
            "shape": tuple(int(dim) for dim in array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
        }
    if _is_torch_tensor(value):
        tensor = value.detach().cpu()
        dtype = str(getattr(tensor, "dtype", "<unknown>"))
        shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
        try:
            array = np.ascontiguousarray(tensor.numpy())
        except (TypeError, RuntimeError):
            array = np.ascontiguousarray(tensor.float().numpy())
        return {
            "kind": "torch",
            "shape": shape,
            "dtype": dtype,
            "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
        }
    encoded = repr(value).encode("utf-8")
    return {
        "kind": type(value).__name__,
        "repr_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _json_normalized(value: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(value, sort_keys=True))


def _serializable_dispatch_record(record: DispatchRecord) -> dict[str, object]:
    return {
        "index": record.index,
        "frame": record.frame,
        "shader": record.shader,
        "dispatch_size": record.dispatch_size,
        "symbols": dict(record.symbols),
        "push_constant_values": dict(record.push_constant_values),
        "descriptor_views": record.descriptor_views,
        "tensor_snapshots": [asdict(snapshot) for snapshot in record.tensor_snapshots],
        "reads": [
            {
                "field": field_name,
                "tensor": tensor.name,
                "writer": None
                if tensor.writer is None
                else {
                    "frame": tensor.writer.frame,
                    "shader": tensor.writer.shader,
                    "dispatch_index": tensor.writer.dispatch_index,
                },
            }
            for field_name, tensor in record.reads
        ],
        "writes": [
            {
                "field": field_name,
                "tensor": tensor.name,
                "writer": None
                if tensor.writer is None
                else {
                    "frame": tensor.writer.frame,
                    "shader": tensor.writer.shader,
                    "dispatch_index": tensor.writer.dispatch_index,
                },
            }
            for field_name, tensor in record.writes
        ],
    }


def _make_output_hook(
    *,
    tensor: LogicalTensor,
    captured: dict[str, object],
):
    def _hook(_module: object, _inputs: object, output: object) -> None:
        probe = tensor.pytorch_probe
        assert probe is not None
        captured[tensor.name] = select_probe_value(
            output,
            index=probe.index,
            selector=probe.selector,
        )

    return _hook


def _make_input_hook(
    *,
    tensor: LogicalTensor,
    captured: dict[str, object],
):
    def _hook(_module: object, inputs: object, _output: object) -> None:
        probe = tensor.pytorch_probe
        assert probe is not None
        captured[tensor.name] = select_probe_value(
            inputs,
            index=probe.index,
            selector=probe.selector,
        )

    return _hook


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "artifact"


def _finite_stat(array: np.ndarray, stat: str) -> float | None:
    numeric = array.astype(np.float64, copy=False)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return None
    if stat == "min":
        return float(finite.min())
    if stat == "max":
        return float(finite.max())
    if stat == "mean":
        return float(finite.mean())
    raise ValueError(f"unknown finite stat {stat}")
