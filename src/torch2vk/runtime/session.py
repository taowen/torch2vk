"""RuntimeSession materialization, dispatch, and lifecycle orchestration."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shutil
import struct
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, TypeGuard

import numpy as np
from vulkan import (
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
)

from torch2vk.checkpoints.checkpoint_tensor import CheckpointTensor
from torch2vk.checkpoints.safetensors import open_safetensors_mmap
from torch2vk.runtime.compare import (
    CompareAssertionError,
    TensorCompareResult,
    as_numpy_array,
    compare_arrays,
    normalize_reference_outputs,
    select_probe_value,
    write_compare_summary,
)
from torch2vk.runtime.frame import FrameContext
from torch2vk.runtime.logical import (
    DispatchWriter,
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    TensorSemantic,
)
from torch2vk.runtime.shader import (
    DTypeReference,
    DispatchRecord,
    DispatchTensorSnapshot,
    PushConstantInput,
    PushConstantSpec,
    PushConstantType,
    ShaderVariant,
    IOKind,
    TensorFieldSpec,
    eval_expr,
)
from torch2vk.vulkan.allocation import BufferAllocation, BufferSlice
from torch2vk.vulkan.compute_pipeline import ComputePipeline, DescriptorBufferBinding
from torch2vk.vulkan.device import VulkanDevice
from torch2vk.vulkan.types import TensorSpec, bind_tensor_layout_symbols, concrete_nbytes

if TYPE_CHECKING:
    import torch


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


_PYTORCH_ARTIFACT_CACHE_VERSION = 1


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


class RuntimeSession:
    """The single runtime owner for LogicalTensor materialization and shader dispatch."""

    def __init__(
        self,
        *,
        device_index: int = 0,
        artifact_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
    ) -> None:
        self.device = VulkanDevice(physical_device_index=device_index)
        self.artifact_dir = Path(
            ".cache/torch2vk/generated" if artifact_dir is None else artifact_dir
        )
        self.model_dir = None if model_dir is None else Path(model_dir).expanduser().resolve()
        self._inputs: dict[LogicalTensor, object] = {}
        self._frame_stack: list[FrameContext] = []
        self._frame_history: dict[str, FrameContext] = {}
        self._dispatch_records: list[DispatchRecord] = []
        self._compare_results: list[TensorCompareResult] = []
        self._pipeline_cache: dict[tuple[object, ...], ComputePipeline] = {}
        self._pytorch_models: dict[object, object] = {}
        self._model_allocations: list[BufferAllocation] = []
        self._request_allocations: list[BufferAllocation] = []
        self._frame_allocations: list[tuple[LogicalTensor, BufferAllocation]] = []
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        device_index: int = 0,
        artifact_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
    ) -> "RuntimeSession":
        return cls(device_index=device_index, artifact_dir=artifact_dir, model_dir=model_dir)

    def __enter__(self) -> "RuntimeSession":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def dispatch_records(self) -> tuple[DispatchRecord, ...]:
        return tuple(self._dispatch_records)

    @property
    def compare_results(self) -> tuple[TensorCompareResult, ...]:
        return tuple(self._compare_results)

    def register_inputs(self, inputs: Mapping[LogicalTensor, object]) -> None:
        for tensor, value in inputs.items():
            if not isinstance(tensor, LogicalTensor):
                raise TypeError(
                    f"register_inputs key must be LogicalTensor, got {type(tensor).__name__}"
                )
            tensor.validate_declaration()
            if tensor.role is not TensorRole.INPUT:
                raise ValueError(f"{tensor.name} is not an input tensor")
            self._inputs[tensor] = value

    def initialize_request_state(self, states: Mapping[LogicalTensor, object]) -> None:
        """Upload initial values for REQUEST_STATE tensors such as KV caches."""
        self._require_open()
        for tensor, value in states.items():
            if not isinstance(tensor, LogicalTensor):
                raise TypeError(
                    f"initialize_request_state key must be LogicalTensor, got {type(tensor).__name__}"
                )
            tensor.validate_declaration()
            if tensor.memory is not MemoryClass.REQUEST_STATE:
                raise ValueError(f"{tensor.name} is not REQUEST_STATE memory")
            array = np.ascontiguousarray(value)
            expected = _tensor_nbytes(tensor.spec)
            if array.nbytes != expected:
                raise ValueError(
                    f"{tensor.name} request state has {array.nbytes} bytes, expected {expected}"
                )
            if expected == 0:
                with tensor.runtime_write_scope():
                    tensor.buffer = None
                    tensor.descriptor_nbytes = 0
                continue
            if tensor.buffer is None:
                ((slice_, allocation),) = self.device.upload_numpy_arrays_with_allocations(
                    [(tensor.name, array)]
                )
                with tensor.runtime_write_scope():
                    tensor.buffer = slice_
                    tensor.descriptor_nbytes = expected
                self._request_allocations.append(allocation)
                continue
            ((source, allocation),) = self.device.upload_numpy_arrays_with_allocations(
                [(tensor.name, array)]
            )
            try:
                self.device.copy_buffer(
                    source.allocation.buffer,
                    tensor.buffer.allocation.buffer,
                    expected,
                    src_offset=source.offset,
                    dst_offset=tensor.buffer.offset,
                )
            finally:
                allocation.close()

    def read_request_state(self, tensor: LogicalTensor) -> np.ndarray:
        """Read back a materialized REQUEST_STATE tensor."""
        self._require_open()
        tensor.validate_declaration()
        if tensor.memory is not MemoryClass.REQUEST_STATE:
            raise ValueError(f"{tensor.name} is not REQUEST_STATE memory")
        return self.readback(tensor)

    def grow_request_state(
        self,
        tensor: LogicalTensor,
        new_shape: Sequence[int],
        *,
        growth: str = "geometric",
    ) -> None:
        """Grow a REQUEST_STATE tensor's logical shape and backing capacity.

        ``tensor.spec.shape`` is the current logical shape. ``tensor.buffer.nbytes`` is the
        physical capacity, which may be larger after geometric growth.
        """
        self._require_open()
        tensor.validate_declaration()
        if tensor.memory is not MemoryClass.REQUEST_STATE:
            raise ValueError(f"{tensor.name} is not REQUEST_STATE memory")
        if tensor.semantic not in {TensorSemantic.KV_CACHE, TensorSemantic.TOKEN}:
            raise ValueError(
                f"{tensor.name} request-state growth is only supported for TOKEN or KV_CACHE tensors"
            )
        if growth != "geometric":
            raise ValueError(f"Unsupported request-state growth strategy: {growth!r}")

        old_shape = _concrete_shape(tensor.spec)
        resolved_new_shape = tuple(int(dim) for dim in new_shape)
        _validate_request_state_growth_shape(
            tensor=tensor,
            old_shape=old_shape,
            new_shape=resolved_new_shape,
        )
        if resolved_new_shape == old_shape:
            return

        old_logical_nbytes = _tensor_nbytes(tensor.spec)
        new_spec = tensor.spec.with_shape(*resolved_new_shape)
        new_logical_nbytes = _tensor_nbytes(new_spec)
        if new_logical_nbytes == 0:
            with tensor.runtime_write_scope():
                tensor.spec = new_spec
                tensor.descriptor_nbytes = 0
                tensor.version += 1
            return
        buffer = tensor.buffer
        if buffer is not None and buffer.nbytes >= new_logical_nbytes:
            with tensor.runtime_write_scope():
                tensor.spec = new_spec
                tensor.descriptor_nbytes = new_logical_nbytes
                tensor.version += 1
            return

        new_capacity_nbytes = _grown_request_state_capacity_nbytes(
            old_capacity_nbytes=0 if buffer is None else buffer.nbytes,
            required_nbytes=new_logical_nbytes,
        )
        new_allocation = self.device.memory_manager.allocate_device_local_buffer(
            new_capacity_nbytes,
            usage_flags=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        )
        old_allocation = None if buffer is None else buffer.allocation
        try:
            if buffer is not None and old_logical_nbytes > 0:
                copy_nbytes = min(
                    old_logical_nbytes,
                    tensor.descriptor_nbytes
                    if tensor.descriptor_nbytes is not None
                    else old_logical_nbytes,
                    buffer.nbytes,
                )
                if copy_nbytes > 0:
                    self.device.copy_buffer(
                        buffer.allocation.buffer,
                        new_allocation.buffer,
                        copy_nbytes,
                        src_offset=buffer.offset,
                        dst_offset=new_allocation.offset,
                    )
            elif old_logical_nbytes > 0:
                raise RuntimeError(
                    f"{tensor.name} cannot grow from non-empty logical shape before it is materialized"
                )
            with tensor.runtime_write_scope():
                tensor.buffer = BufferSlice(
                    allocation=new_allocation,
                    offset=new_allocation.offset,
                    nbytes=new_capacity_nbytes,
                )
                tensor.spec = new_spec
                tensor.descriptor_nbytes = new_logical_nbytes
                tensor.version += 1
            self._request_allocations.append(new_allocation)
        except Exception:
            new_allocation.close()
            raise
        if old_allocation is not None:
            self._release_request_allocation(old_allocation)

    def release_request_state(self, tensors: Sequence[LogicalTensor] | None = None) -> None:
        """Release selected request-state buffers, or all request allocations when omitted."""
        self._require_open()
        if tensors is None:
            for allocation in reversed(self._request_allocations):
                allocation.close()
            self._request_allocations.clear()
            return
        requested = set(tensors)
        retained: list[BufferAllocation] = []
        released: list[BufferAllocation] = []
        for allocation in self._request_allocations:
            if any(
                tensor.buffer is not None and tensor.buffer.allocation is allocation
                for tensor in requested
            ):
                allocation.close()
                released.append(allocation)
            else:
                retained.append(allocation)
        self._request_allocations = retained
        for tensor in requested:
            if tensor.buffer is not None and any(
                tensor.buffer.allocation is allocation for allocation in released
            ):
                with tensor.runtime_write_scope():
                    tensor.buffer = None
                    tensor.descriptor_nbytes = None
                    tensor.writer = None

    @contextmanager
    def frame(
        self,
        name: str,
        *,
        pytorch_model: object | None = None,
        pytorch_model_class: object | None = None,
        pytorch_model_submodule: str | None = None,
        pytorch_args: tuple[object, ...] = (),
        pytorch_kwargs: Mapping[str, object] | None = None,
        reference_model: object | None = None,
    ):
        if not name:
            raise ValueError("frame name must be non-empty")
        if pytorch_model is None and pytorch_model_class is not None:
            loaded = self._load_pytorch_model(pytorch_model_class)
            if loaded is not None and pytorch_model_submodule:
                for attr in pytorch_model_submodule.split("."):
                    loaded = getattr(loaded, attr)
            pytorch_model = loaded
        context = FrameContext(
            frame=name,
            start_dispatch_index=len(self._dispatch_records),
            pytorch_model=pytorch_model,
            pytorch_args=tuple(pytorch_args),
            pytorch_kwargs={} if pytorch_kwargs is None else dict(pytorch_kwargs),
            reference_model=reference_model,
        )
        self._frame_stack.append(context)
        candidate_completed = False
        try:
            yield context
            candidate_completed = True
        finally:
            try:
                if candidate_completed:
                    self._compare_frame(context)
            finally:
                popped = self._frame_stack.pop()
                if popped is not context:
                    raise RuntimeError("RuntimeSession frame stack corrupted")
                self._frame_history[context.frame] = context
                self._release_frame_allocations()

    def dispatch(self, variant: ShaderVariant, **arguments: object) -> None:
        self._require_open()
        frame = self._current_frame()
        contract = variant.contract
        expected = {field.name for field in contract.fields}
        provided = set(arguments)
        if missing := expected - provided:
            raise ValueError(f"{variant.name} missing tensor fields: {sorted(missing)}")
        if extra := provided - expected:
            raise ValueError(f"{variant.name} got unexpected fields: {sorted(extra)}")

        tensors: dict[str, LogicalTensor] = {}
        for name, argument in arguments.items():
            if not isinstance(argument, LogicalTensor):
                raise TypeError(
                    f"{variant.name}.{name} expects LogicalTensor, got {type(argument).__name__}"
                )
            argument.validate_declaration()
            tensors[name] = argument
        frame.used_tensors.extend(tensors[field.name] for field in contract.fields)

        symbols = self._bind_shape_symbols(contract.fields, tensors)
        for field in contract.input_fields:
            self._materialize_read(tensors[field.name])
        for field in contract.output_fields:
            self._materialize_write(tensors[field.name], io_kind=field.io_kind)

        descriptor_views = tuple(
            _descriptor_view_for_field(field, tensors[field.name]) for field in contract.fields
        )
        push_constants, push_values = self._pack_push_constants(
            contract.push_constants,
            tensors=tensors,
            symbols=symbols,
        )
        dispatch_size = (
            eval_expr(contract.dispatch[0], symbols),
            eval_expr(contract.dispatch[1], symbols),
            eval_expr(contract.dispatch[2], symbols),
        )
        if any(dim <= 0 for dim in dispatch_size):
            raise ValueError(f"{variant.name} resolved non-positive dispatch {dispatch_size}")

        pipeline = self._pipeline_for_variant(variant)
        pipeline.dispatch(
            buffers=[view for _, view in descriptor_views],
            group_count_x=dispatch_size[0],
            group_count_y=dispatch_size[1],
            group_count_z=dispatch_size[2],
            push_constants=push_constants,
        )

        index = len(self._dispatch_records)
        record = DispatchRecord(
            index=index,
            frame=frame.frame,
            shader=variant.name,
            reads=tuple((field.name, tensors[field.name]) for field in contract.input_fields),
            writes=tuple((field.name, tensors[field.name]) for field in contract.output_fields),
            logical_reads=tuple(
                (field.name, tensors[field.name].name) for field in contract.input_fields
            ),
            logical_writes=tuple(
                (field.name, tensors[field.name].name) for field in contract.output_fields
            ),
            symbols=tuple(sorted(symbols.items())),
            dispatch_size=dispatch_size,
            descriptor_views=tuple(
                _record_descriptor_view(index, field, tensors[field.name])
                for index, field in enumerate(contract.fields)
            ),
            tensor_snapshots=tuple(
                _record_tensor_snapshot(field, tensors[field.name]) for field in contract.fields
            ),
            push_constant_values=tuple(sorted(push_values.items())),
        )
        self._dispatch_records.append(record)
        for field in contract.output_fields:
            tensor = tensors[field.name]
            with tensor.runtime_write_scope():
                tensor.version += 1
                tensor.writer = DispatchWriter(
                    frame=frame.frame,
                    shader=variant.name,
                    dispatch_index=index,
                )
            frame.written_tensors.append(tensor)

    def readback(self, tensor: LogicalTensor) -> np.ndarray:
        self._require_open()
        if tensor.buffer is None:
            if _tensor_nbytes(tensor.spec) == 0:
                return self.device.empty_tensor(spec=tensor.spec)
            raise RuntimeError(f"{tensor.name} is not materialized")
        return self.device.readback_tensor(
            spec=tensor.spec, slice=tensor.buffer, layout=tensor.layout
        )

    def debug_materialization(self, tensor: LogicalTensor) -> BufferSlice | None:
        return tensor.buffer

    def _compare_frame(self, frame: FrameContext) -> None:
        written = _unique_tensors(frame.written_tensors)
        if frame.reference_model is not None:
            reference_targets = _default_frame_compare_targets(written)
            if not reference_targets:
                return
            reference_outputs = normalize_reference_outputs(
                _call_reference_model(
                    frame.reference_model,
                    inputs=self._inputs,
                    frame=frame.frame,
                )
            )
            for tensor in reference_targets:
                try:
                    expected = reference_outputs[tensor.name]
                except KeyError as exc:
                    raise KeyError(
                        f"Reference model did not return artifact for {tensor.name}"
                    ) from exc
                self._record_compare(
                    tensor=tensor,
                    frame=frame,
                    candidate=self.readback(tensor),
                    expected=expected,
                )
            return

        if frame.pytorch_model is None:
            return

        probe_targets = _default_frame_compare_targets(
            tensor for tensor in written if tensor.pytorch_probe is not None
        )
        if not probe_targets:
            return
        expected_by_tensor, missing_probe_targets = self._load_cached_pytorch_artifacts(
            frame,
            probe_targets,
            frame.pytorch_model,
        )
        if missing_probe_targets:
            captured = self._run_pytorch_probes(frame, missing_probe_targets)
            expected_by_tensor.update(captured)
            self._store_cached_pytorch_artifacts(
                frame,
                missing_probe_targets,
                frame.pytorch_model,
                captured,
            )
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
        expected_by_tensor, missing_probe_targets = self._load_cached_pytorch_artifacts(
            frame,
            probe_targets,
            frame.pytorch_model,
        )
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
            "inputs": self._frame_input_fingerprints(frame.frame),
            "explicit_pytorch_args": [
                self._pytorch_value_fingerprint(value) for value in frame.pytorch_args
            ],
            "explicit_pytorch_kwargs": {
                key: self._pytorch_value_fingerprint(value)
                for key, value in sorted(frame.pytorch_kwargs.items())
            },
            "model": self._model_fingerprint(model),
        }

    def _frame_input_fingerprints(self, frame_name: str) -> list[dict[str, object]]:
        prefix = f"{frame_name}."
        entries: list[dict[str, object]] = []
        for tensor, value in self._inputs.items():
            if tensor.role is not TensorRole.INPUT or not tensor.name.startswith(prefix):
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

        args, kwargs = self._pytorch_forward_inputs(frame, model)
        try:
            output = module_like(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()

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
        return captured

    def _pytorch_forward_inputs(
        self,
        frame: FrameContext,
        model: object,
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        if frame.pytorch_args or frame.pytorch_kwargs:
            return (
                tuple(self._resolve_pytorch_value(value) for value in frame.pytorch_args),
                {
                    key: self._resolve_pytorch_value(value)
                    for key, value in frame.pytorch_kwargs.items()
                },
            )
        return (), self._infer_pytorch_kwargs(frame=frame, model=model)

    def _infer_pytorch_kwargs(self, *, frame: FrameContext, model: object) -> dict[str, object]:
        import torch

        forward = getattr(model, "forward", model)
        if not callable(forward):
            raise TypeError(f"PyTorch model forward is not callable: {type(model).__name__}")
        signature = inspect.signature(forward)
        parameter_names = {
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        device = self._pytorch_model_device(model)
        prefix = f"{frame.frame}."
        kwargs: dict[str, object] = {}
        for tensor, value in self._inputs.items():
            if tensor.role is not TensorRole.INPUT or not tensor.name.startswith(prefix):
                continue
            kwarg = tensor.name.removeprefix(prefix)
            if kwarg in parameter_names:
                pytorch_value = self._as_pytorch_input(value)
                if isinstance(pytorch_value, torch.Tensor):
                    pytorch_value = pytorch_value.to(device)
                kwargs[kwarg] = pytorch_value
        return kwargs

    def _resolve_pytorch_value(self, value: object) -> object:
        if isinstance(value, LogicalTensor):
            try:
                registered = self._inputs[value]
            except KeyError as exc:
                raise RuntimeError(
                    f"{value.name} is used as a PyTorch input but was not registered"
                ) from exc
            return self._as_pytorch_input(registered)
        return self._as_pytorch_input(value)

    def _as_pytorch_input(self, value: object) -> object:
        import torch

        if _is_torch_tensor(value):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(value))
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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release_frame_allocations()
        for pipeline in self._pipeline_cache.values():
            pipeline.close()
        self._pipeline_cache.clear()
        for allocation in reversed(self._request_allocations):
            allocation.close()
        self._request_allocations.clear()
        for allocation in reversed(self._model_allocations):
            allocation.close()
        self._model_allocations.clear()
        self.device.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("RuntimeSession is closed")

    def _current_frame(self) -> FrameContext:
        if not self._frame_stack:
            raise RuntimeError("RuntimeSession.dispatch requires an active rt.frame(...)")
        return self._frame_stack[-1]

    def _bind_shape_symbols(
        self,
        fields: tuple[TensorFieldSpec, ...],
        tensors: Mapping[str, LogicalTensor],
    ) -> dict[str, int]:
        symbols: dict[str, int] = {}
        dtype_by_field: dict[str, str] = {}
        for field in fields:
            tensor = tensors[field.name]
            contract = field.contract
            dtype = contract.dtype
            if isinstance(dtype, DTypeReference):
                if dtype.field_name not in dtype_by_field:
                    raise ValueError(
                        f"{field.name} references unknown dtype field {dtype.field_name}"
                    )
                expected_dtypes = (dtype_by_field[dtype.field_name],)
            elif isinstance(dtype, tuple):
                expected_dtypes = dtype
            else:
                expected_dtypes = (dtype,)
            if tensor.spec.dtype not in expected_dtypes:
                raise ValueError(
                    f"{field.name} expects dtype {expected_dtypes}, got {tensor.spec.dtype} "
                    f"from {tensor.name}"
                )
            dtype_by_field[field.name] = tensor.spec.dtype
            if len(tensor.spec.shape) != len(contract.shape):
                raise ValueError(
                    f"{field.name} expects rank {len(contract.shape)}, got {len(tensor.spec.shape)} "
                    f"from {tensor.name}"
                )
            for expected_dim, actual_dim in zip(contract.shape, tensor.spec.shape, strict=True):
                if not isinstance(actual_dim, int):
                    raise ValueError(f"{tensor.name} has unresolved shape {tensor.spec.shape}")
                if isinstance(expected_dim, int):
                    if actual_dim != expected_dim:
                        raise ValueError(
                            f"{field.name} expects dim {expected_dim}, got {actual_dim}"
                        )
                elif isinstance(expected_dim, str):
                    previous = symbols.get(expected_dim)
                    if previous is None:
                        symbols[expected_dim] = actual_dim
                    elif previous != actual_dim:
                        raise ValueError(
                            f"Shape symbol {expected_dim} bound to both {previous} and {actual_dim}"
                        )
                else:
                    expected_value = eval_expr(expected_dim, symbols)
                    if actual_dim != expected_value:
                        raise ValueError(
                            f"{field.name} expects dim {expected_value}, got {actual_dim}"
                        )
            bind_tensor_layout_symbols(contract.layout, tensor.layout, symbols)
        return symbols

    def _materialize_read(self, tensor: LogicalTensor) -> None:
        if tensor.buffer is not None:
            return
        if tensor.role is TensorRole.WEIGHT:
            self._materialize_weight(tensor)
            return
        if tensor.role is TensorRole.INPUT:
            self._materialize_input(tensor)
            return
        raise RuntimeError(f"{tensor.name} cannot be read before it is materialized or written")

    def _materialize_write(self, tensor: LogicalTensor, *, io_kind: IOKind) -> None:
        if io_kind is IOKind.INOUT and tensor.buffer is not None:
            return
        size = _tensor_nbytes(tensor.spec)
        if size == 0:
            with tensor.runtime_write_scope():
                tensor.buffer = None
                tensor.descriptor_nbytes = 0
            return
        if tensor.memory is MemoryClass.HOST_OUTPUT:
            allocation = self.device.allocate_host_visible_allocation(size)
        elif tensor.memory in {MemoryClass.FRAME_WORKSPACE, MemoryClass.OP_SCRATCH}:
            allocation = self.device.memory_manager.allocate_device_local_buffer(
                size,
                usage_flags=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            )
        elif tensor.memory is MemoryClass.REQUEST_STATE:
            allocation = self.device.memory_manager.allocate_device_local_buffer(
                size,
                usage_flags=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            )
            self._request_allocations.append(allocation)
        else:
            raise ValueError(
                f"{tensor.name} cannot be materialized for write with memory={tensor.memory}"
            )
        with tensor.runtime_write_scope():
            tensor.buffer = BufferSlice(
                allocation=allocation, offset=allocation.offset, nbytes=size
            )
            tensor.descriptor_nbytes = size
        if tensor.lifetime in {TensorLifetime.FRAME, TensorLifetime.OP}:
            self._frame_allocations.append((tensor, allocation))

    def _materialize_weight(self, tensor: LogicalTensor) -> None:
        tensor.validate_declaration()
        checkpoint = self._resolve_weight_checkpoint(tensor)
        with open_safetensors_mmap(checkpoint) as storage:
            checkpoint_tensor = CheckpointTensor.open(
                storage=storage,
                tensor_key=tensor.name,
                dtype=tensor.spec.dtype,
                shape=tensor.concrete_shape,
                layout=tensor.layout,
            )
            ((slice_, allocation),) = self.device.upload_checkpoint_tensors_with_allocations(
                [(tensor.name, checkpoint_tensor)]
            )
        with tensor.runtime_write_scope():
            tensor.buffer = slice_
            tensor.descriptor_nbytes = slice_.nbytes
        self._model_allocations.append(allocation)

    def _resolve_weight_checkpoint(self, tensor: LogicalTensor) -> Path:
        if self.model_dir is None:
            raise RuntimeError(
                f"{tensor.name} is a weight tensor but RuntimeSession has no model_dir to resolve checkpoint"
            )
        primary = self.model_dir / "model.safetensors"
        if primary.is_file():
            return primary
        index = self.model_dir / "model.safetensors.index.json"
        if index.is_file():
            return index
        candidates = sorted(self.model_dir.glob("*.safetensors")) + sorted(
            self.model_dir.glob("*.safetensors.index.json")
        )
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise RuntimeError(
                f"{tensor.name} could not find a safetensors checkpoint in {self.model_dir}"
            )
        raise RuntimeError(
            f"{tensor.name} found multiple safetensors checkpoints in {self.model_dir}; "
            "use model.safetensors or model.safetensors.index.json as the canonical checkpoint"
        )

    def _materialize_input(self, tensor: LogicalTensor) -> None:
        if tensor not in self._inputs:
            raise RuntimeError(f"{tensor.name} requires missing input")
        value = self._inputs[tensor]
        array = np.ascontiguousarray(value)
        expected = _tensor_nbytes(tensor.spec)
        if array.nbytes != expected:
            raise ValueError(f"{tensor.name} input has {array.nbytes} bytes, expected {expected}")
        if tensor.memory is MemoryClass.HOST_INPUT:
            allocation = self.device.allocate_host_visible_allocation(expected)
            allocation.buffer.write_bytes_at(allocation.offset, memoryview(array).cast("B"))
            self.device.memory_manager.host_upload_ring.flush(allocation=allocation)
        else:
            ((slice_, allocation),) = self.device.upload_numpy_arrays_with_allocations(
                [(tensor.name, array)]
            )
            with tensor.runtime_write_scope():
                tensor.buffer = slice_
                tensor.descriptor_nbytes = slice_.nbytes
            self._request_allocations.append(allocation)
            return
        with tensor.runtime_write_scope():
            tensor.buffer = BufferSlice(
                allocation=allocation, offset=allocation.offset, nbytes=expected
            )
            tensor.descriptor_nbytes = expected
        if tensor.lifetime in {TensorLifetime.FRAME, TensorLifetime.OP}:
            self._frame_allocations.append((tensor, allocation))

    def _pack_push_constants(
        self,
        spec: PushConstantSpec | None,
        *,
        tensors: Mapping[str, LogicalTensor],
        symbols: Mapping[str, int],
    ) -> tuple[bytes | None, dict[str, int | float]]:
        if spec is None:
            return None, {}
        data = bytearray(spec.size)
        values: dict[str, int | float] = {}
        for field in spec.fields:
            raw = field.value
            if isinstance(raw, PushConstantInput):
                raise ValueError(f"PushConstantInput {raw.name!r} is not supported by this MVP")
            if callable(raw):
                value = raw(tensors, symbols)
            elif isinstance(raw, str):
                value = symbols[raw]
            elif isinstance(raw, int | float):
                value = raw
            else:
                value = eval_expr(raw, symbols)
            values[field.name] = value
            data[field.offset : field.offset + field.size] = _pack_push_constant_value(
                field.dtype, value
            )
        return bytes(data), values

    def _pipeline_for_variant(self, variant: ShaderVariant) -> ComputePipeline:
        spv_path = self._spv_path_for_variant(variant)
        descriptor_count = len(variant.contract.fields)
        key = (
            str(spv_path),
            descriptor_count,
            variant.specialization_constants,
            0 if variant.contract.push_constants is None else variant.contract.push_constants.size,
            variant.execution_requirements,
        )
        pipeline = self._pipeline_cache.get(key)
        if pipeline is not None:
            return pipeline
        pipeline = ComputePipeline(
            self.device,
            shader_spv_path=spv_path,
            storage_buffer_count=descriptor_count,
            specialization_constants=None
            if variant.specialization_constants is None
            else dict(variant.specialization_constants),
            push_constant_size=0
            if variant.contract.push_constants is None
            else variant.contract.push_constants.size,
            execution_requirements=variant.execution_requirements,
        )
        self._pipeline_cache[key] = pipeline
        return pipeline

    def _spv_path_for_variant(self, variant: ShaderVariant) -> Path:
        if variant.precompiled_spv_path is not None:
            return variant.precompiled_spv_path
        compiler = shutil.which("glslc")
        if compiler is None:
            raise RuntimeError("glslc is required to compile inline ShaderVariant source")
        compile_args = (
            "-fshader-stage=compute",
            "--target-env=vulkan1.3",
            "-O",
            "-g",
        )
        source_hash = hashlib.sha256(
            "\n".join(
                (
                    variant.source,
                    repr(compile_args),
                    repr(variant.include_dirs),
                    repr(variant.compile_defines),
                )
            ).encode("utf-8")
        ).hexdigest()[:16]
        stem = f"{variant.name}.{source_hash}"
        glsl_path = self.artifact_dir / f"{stem}.comp"
        spv_path = self.artifact_dir / f"{stem}.spv"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if not spv_path.is_file():
            glsl_path.write_text(variant.source, encoding="utf-8")
            include_args: list[str | Path] = []
            for include_dir in variant.include_dirs:
                include_args.extend(("-I", include_dir))
            define_args = [f"-D{define}" for define in variant.compile_defines]
            subprocess.run(
                [
                    compiler,
                    *compile_args,
                    *include_args,
                    *define_args,
                    str(glsl_path),
                    "-o",
                    str(spv_path),
                ],
                check=True,
                cwd=str(glsl_path.parent),
            )
        return spv_path

    def _release_frame_allocations(self) -> None:
        while self._frame_allocations:
            tensor, allocation = self._frame_allocations.pop()
            if tensor.buffer is not None and tensor.buffer.allocation is allocation:
                with tensor.runtime_write_scope():
                    tensor.buffer = None
                    tensor.descriptor_nbytes = None
            allocation.close()

    def _release_request_allocation(self, allocation: BufferAllocation) -> None:
        self._request_allocations = [
            request_allocation
            for request_allocation in self._request_allocations
            if request_allocation is not allocation
        ]
        allocation.close()


def _tensor_nbytes(spec: TensorSpec) -> int:
    concrete_shape: list[int] = []
    for dim in spec.shape:
        if not isinstance(dim, int):
            raise ValueError(f"Expected concrete tensor shape, got {spec.shape}")
        concrete_shape.append(dim)
    return concrete_nbytes(dtype=spec.dtype, shape=tuple(concrete_shape))


def _concrete_shape(spec: TensorSpec) -> tuple[int, ...]:
    concrete_shape: list[int] = []
    for dim in spec.shape:
        if not isinstance(dim, int):
            raise ValueError(f"Expected concrete tensor shape, got {spec.shape}")
        concrete_shape.append(dim)
    return tuple(concrete_shape)


def _validate_request_state_growth_shape(
    *,
    tensor: LogicalTensor,
    old_shape: tuple[int, ...],
    new_shape: tuple[int, ...],
) -> None:
    if len(new_shape) != len(old_shape):
        raise ValueError(
            f"{tensor.name} grow_request_state cannot change rank from "
            f"{len(old_shape)} to {len(new_shape)}"
        )
    if any(dim < 0 for dim in new_shape):
        raise ValueError(f"{tensor.name} grow_request_state shape must be non-negative")
    shrinking = [
        (index, old_dim, new_dim)
        for index, (old_dim, new_dim) in enumerate(zip(old_shape, new_shape, strict=True))
        if new_dim < old_dim
    ]
    if shrinking:
        index, old_dim, new_dim = shrinking[0]
        raise ValueError(
            f"{tensor.name} grow_request_state cannot shrink dim {index} "
            f"from {old_dim} to {new_dim}"
        )
    if tensor.semantic is TensorSemantic.KV_CACHE:
        if len(old_shape) != 4:
            raise ValueError(f"{tensor.name} KV_CACHE growth expects rank 4, got {len(old_shape)}")
        _require_only_growth_dim(tensor=tensor, old_shape=old_shape, new_shape=new_shape, dim=2)
        return
    if tensor.semantic is TensorSemantic.TOKEN:
        _require_only_growth_dim(
            tensor=tensor,
            old_shape=old_shape,
            new_shape=new_shape,
            dim=len(old_shape) - 1,
        )


def _require_only_growth_dim(
    *,
    tensor: LogicalTensor,
    old_shape: tuple[int, ...],
    new_shape: tuple[int, ...],
    dim: int,
) -> None:
    for index, (old_dim, new_dim) in enumerate(zip(old_shape, new_shape, strict=True)):
        if index != dim and old_dim != new_dim:
            raise ValueError(
                f"{tensor.name} grow_request_state can only change dim {dim}; "
                f"dim {index} changed from {old_dim} to {new_dim}"
            )


def _grown_request_state_capacity_nbytes(
    *,
    old_capacity_nbytes: int,
    required_nbytes: int,
) -> int:
    if required_nbytes <= 0:
        return 0
    if old_capacity_nbytes <= 0:
        return required_nbytes
    return max(required_nbytes, old_capacity_nbytes * 2)


def _descriptor_view_for_field(
    field: TensorFieldSpec,
    tensor: LogicalTensor,
) -> tuple[TensorFieldSpec, DescriptorBufferBinding]:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} is not materialized")
    return field, DescriptorBufferBinding.from_slice(
        tensor.buffer,
        descriptor_nbytes=tensor.descriptor_nbytes,
    )


def _record_descriptor_view(
    index: int, field: TensorFieldSpec, tensor: LogicalTensor
) -> tuple[str, int, int, int]:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} is not materialized")
    return (
        field.name,
        index,
        tensor.buffer.offset,
        tensor.descriptor_nbytes or 0,
    )


def _record_tensor_snapshot(
    field: TensorFieldSpec, tensor: LogicalTensor
) -> DispatchTensorSnapshot:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} is not materialized")
    return DispatchTensorSnapshot(
        field=field.name,
        tensor=tensor.name,
        shape=_concrete_shape(tensor.spec),
        dtype=tensor.spec.dtype,
        descriptor_offset=tensor.buffer.offset,
        descriptor_nbytes=tensor.descriptor_nbytes or 0,
        version=tensor.version,
    )


def _pack_push_constant_value(dtype: PushConstantType, value: int | float) -> bytes:
    if dtype is PushConstantType.UINT32:
        integer = int(value)
        if integer < 0 or integer > 0xFFFF_FFFF:
            raise ValueError(f"uint32 push constant out of range: {integer}")
        return struct.pack("<I", integer)
    if dtype is PushConstantType.INT32:
        integer = int(value)
        if integer < -(2**31) or integer > 2**31 - 1:
            raise ValueError(f"int32 push constant out of range: {integer}")
        return struct.pack("<i", integer)
    if dtype is PushConstantType.UINT64:
        integer = int(value)
        if integer < 0 or integer > 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError(f"uint64 push constant out of range: {integer}")
        return struct.pack("<Q", integer)
    if dtype is PushConstantType.FLOAT32:
        return struct.pack("<f", float(value))
    raise TypeError(f"Unsupported push constant dtype {dtype}")


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


def _call_reference_model(
    reference_model: object,
    *,
    inputs: Mapping[LogicalTensor, object],
    frame: str,
) -> Mapping[object, object]:
    if not callable(reference_model):
        raise TypeError(f"reference_model must be callable, got {type(reference_model).__name__}")
    signature = inspect.signature(reference_model)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        and parameter.default is inspect.Parameter.empty
    ]
    if not parameters:
        result = reference_model()
    elif len(parameters) == 1:
        result = reference_model(dict(inputs))
    else:
        result = reference_model(dict(inputs), frame)
    if not isinstance(result, Mapping):
        raise TypeError(f"reference_model must return a Mapping, got {type(result).__name__}")
    return result


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
