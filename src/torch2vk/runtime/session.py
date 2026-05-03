"""RuntimeSession materialization, dispatch, and lifecycle orchestration."""

from __future__ import annotations

import hashlib
import inspect
import shutil
import struct
import subprocess
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, cast

import numpy as np
from vulkan import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT

from torch2vk.checkpoints.checkpoint_tensor import CheckpointTensor
from torch2vk.checkpoints.safetensors import open_safetensors_mmap
from torch2vk.runtime.compare import TensorCompareResult, compare_arrays, normalize_reference_outputs, select_probe_value
from torch2vk.runtime.frame import FrameContext, FrameScope
from torch2vk.runtime.logical import (
    DispatchWriter,
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
)
from torch2vk.runtime.shader import (
    DTypeReference,
    DispatchRecord,
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


class _PyTorchModuleLike(Protocol):
    def named_modules(self) -> Iterable[tuple[str, object]]: ...

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _HookHandleLike(Protocol):
    def remove(self) -> None: ...


class _ForwardHookModuleLike(Protocol):
    def register_forward_hook(self, hook: object) -> _HookHandleLike: ...


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
        self.artifact_dir = Path(".cache/torch2vk/generated" if artifact_dir is None else artifact_dir)
        self.model_dir = None if model_dir is None else Path(model_dir).expanduser().resolve()
        self._inputs: dict[LogicalTensor, object] = {}
        self._frame_stack: list[FrameContext] = []
        self._dispatch_records: list[DispatchRecord] = []
        self._compare_results: list[TensorCompareResult] = []
        self._pipeline_cache: dict[tuple[Any, ...], ComputePipeline] = {}
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
                raise TypeError(f"register_inputs key must be LogicalTensor, got {type(tensor).__name__}")
            tensor.validate_declaration()
            if tensor.role is not TensorRole.INPUT:
                raise ValueError(f"{tensor.name} is not an input tensor")
            self._inputs[tensor] = value

    @contextmanager
    def frame(
        self,
        name: str,
        *,
        scope: Mapping[str, str | int] | None = None,
        dependencies: Iterable[LogicalTensor] = (),
        pytorch_model: object | None = None,
        pytorch_args: tuple[object, ...] = (),
        pytorch_kwargs: Mapping[str, object] | None = None,
        reference_model: object | None = None,
    ):
        if not name:
            raise ValueError("frame name must be non-empty")
        frame_dependencies = tuple(dependencies)
        context = FrameContext(
            scope=FrameScope(frame=name, values={} if scope is None else dict(scope)),
            start_dispatch_index=len(self._dispatch_records),
            dependencies=frame_dependencies,
            pytorch_model=pytorch_model,
            pytorch_args=tuple(pytorch_args),
            pytorch_kwargs={} if pytorch_kwargs is None else dict(pytorch_kwargs),
            reference_model=reference_model,
        )
        self._frame_stack.append(context)
        candidate_completed = False
        try:
            self._materialize_frame_dependencies(frame_dependencies)
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
                raise TypeError(f"{variant.name}.{name} expects LogicalTensor, got {type(argument).__name__}")
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
            frame=frame.scope.frame,
            scope_values=tuple(sorted(frame.scope.values.items())),
            shader=variant.name,
            reads=tuple((field.name, tensors[field.name]) for field in contract.input_fields),
            writes=tuple((field.name, tensors[field.name]) for field in contract.output_fields),
            logical_reads=tuple((field.name, tensors[field.name].name) for field in contract.input_fields),
            logical_writes=tuple((field.name, tensors[field.name].name) for field in contract.output_fields),
            symbols=tuple(sorted(symbols.items())),
            dispatch_size=dispatch_size,
            descriptor_views=tuple(
                _record_descriptor_view(index, field, tensors[field.name])
                for index, field in enumerate(contract.fields)
            ),
            push_constant_values=tuple(sorted(push_values.items())),
        )
        self._dispatch_records.append(record)
        for field in contract.output_fields:
            tensor = tensors[field.name]
            with tensor.runtime_write_scope():
                tensor.version += 1
                tensor.writer = DispatchWriter(
                    frame=frame.scope.frame,
                    shader=variant.name,
                    dispatch_index=index,
                )
            frame.written_tensors.append(tensor)

    def readback(self, tensor: LogicalTensor) -> np.ndarray:
        self._require_open()
        if tensor.buffer is None:
            raise RuntimeError(f"{tensor.name} is not materialized")
        return self.device.readback_tensor(spec=tensor.spec, slice=tensor.buffer, layout=tensor.layout)

    def debug_materialization(self, tensor: LogicalTensor) -> BufferSlice | None:
        return tensor.buffer

    def _compare_frame(self, frame: FrameContext) -> None:
        written = _unique_tensors(frame.written_tensors)
        reference_targets = [tensor for tensor in written if tensor.compare is not None]
        if frame.reference_model is not None and reference_targets:
            reference_outputs = normalize_reference_outputs(
                _call_reference_model(
                    frame.reference_model,
                    inputs=self._inputs,
                    scope=frame.scope,
                )
            )
            for tensor in reference_targets:
                try:
                    expected = reference_outputs[tensor.name]
                except KeyError as exc:
                    raise KeyError(f"Reference model did not return artifact for {tensor.name}") from exc
                self._compare_results.append(
                    compare_arrays(
                        tensor=tensor,
                        scope=frame.scope,
                        candidate=self.readback(tensor),
                        expected=expected,
                    )
                )
            return

        if frame.pytorch_model is None:
            return

        probe_targets = [
            tensor for tensor in reference_targets if tensor.pytorch_probe is not None
        ]
        if not probe_targets:
            return
        expected_by_tensor = self._run_pytorch_probes(frame, probe_targets)
        for tensor in probe_targets:
            try:
                expected = expected_by_tensor[tensor.name]
            except KeyError as exc:
                raise KeyError(f"PyTorch probe did not capture artifact for {tensor.name}") from exc
            self._compare_results.append(
                compare_arrays(
                    tensor=tensor,
                    scope=frame.scope,
                    candidate=self.readback(tensor),
                    expected=expected,
                )
            )

    def _run_pytorch_probes(
        self,
        frame: FrameContext,
        tensors: list[LogicalTensor],
    ) -> dict[str, object]:
        model = frame.pytorch_model
        if model is None:
            raise RuntimeError("PyTorch probe requested without pytorch_model")
        module_like = cast(_PyTorchModuleLike, model)
        modules = dict(module_like.named_modules()) if hasattr(model, "named_modules") else {}
        captured: dict[str, object] = {}
        hooks: list[_HookHandleLike] = []
        root_output_tensors: list[LogicalTensor] = []

        for tensor in tensors:
            probe = tensor.pytorch_probe
            if probe is None:
                continue
            if probe.transform is not None:
                raise NotImplementedError(f"{tensor.name} probe transform is not implemented: {probe.transform}")
            if probe.kind == "derived":
                raise NotImplementedError(f"{tensor.name} derived PyTorchProbe is not implemented")
            if probe.target == "":
                root_output_tensors.append(tensor)
                continue
            try:
                module = cast(_ForwardHookModuleLike, modules[probe.target])
            except KeyError as exc:
                raise KeyError(f"PyTorch module probe target not found: {probe.target}") from exc
            if probe.kind == "module_output":
                hooks.append(
                    module.register_forward_hook(
                        _make_output_hook(tensor=tensor, captured=captured)
                    )
                )
            elif probe.kind == "module_input":
                hooks.append(
                    module.register_forward_hook(
                        _make_input_hook(tensor=tensor, captured=captured)
                    )
                )
            else:
                raise NotImplementedError(f"{tensor.name} unsupported PyTorchProbe kind: {probe.kind}")

        try:
            output = module_like(*frame.pytorch_args, **dict(frame.pytorch_kwargs))
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
                    raise ValueError(f"{field.name} references unknown dtype field {dtype.field_name}")
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
                        raise ValueError(f"{field.name} expects dim {expected_dim}, got {actual_dim}")
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
                        raise ValueError(f"{field.name} expects dim {expected_value}, got {actual_dim}")
            bind_tensor_layout_symbols(contract.layout, tensor.layout, symbols)
        return symbols

    def _materialize_read(self, tensor: LogicalTensor) -> None:
        if tensor.buffer is not None:
            return
        if tensor.source is not None:
            raise RuntimeError(
                f"{tensor.name} is a weight tensor but was not preloaded at frame enter; "
                "include it in rt.frame(..., dependencies=...)"
            )
        if tensor.role is TensorRole.INPUT:
            self._materialize_input(tensor)
            return
        raise RuntimeError(f"{tensor.name} cannot be read before it is materialized or written")

    def _materialize_frame_dependencies(self, dependencies: Iterable[LogicalTensor]) -> None:
        for tensor in dependencies:
            if not isinstance(tensor, LogicalTensor):
                raise TypeError(f"frame dependencies must be LogicalTensor, got {type(tensor).__name__}")
            tensor.validate_declaration()
            if tensor.source is not None and tensor.buffer is None:
                self._materialize_weight(tensor)

    def _materialize_write(self, tensor: LogicalTensor, *, io_kind: IOKind) -> None:
        if io_kind is IOKind.INOUT and tensor.buffer is not None:
            return
        size = _tensor_nbytes(tensor.spec)
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
            raise ValueError(f"{tensor.name} cannot be materialized for write with memory={tensor.memory}")
        with tensor.runtime_write_scope():
            tensor.buffer = BufferSlice(allocation=allocation, offset=allocation.offset, nbytes=size)
            tensor.descriptor_nbytes = size
        if tensor.lifetime in {TensorLifetime.FRAME, TensorLifetime.OP}:
            self._frame_allocations.append((tensor, allocation))

    def _materialize_weight(self, tensor: LogicalTensor) -> None:
        assert tensor.source is not None
        checkpoint = Path(tensor.source.checkpoint)
        if not checkpoint.is_absolute():
            if self.model_dir is None:
                raise RuntimeError(f"{tensor.name} uses relative checkpoint {checkpoint} without model_dir")
            checkpoint = self.model_dir / checkpoint
        with open_safetensors_mmap(checkpoint) as storage:
            checkpoint_tensor = CheckpointTensor.open(
                storage=storage,
                tensor_key=tensor.source.key,
                dtype=tensor.source.dtype,
                shape=tensor.source.shape,
                layout=tensor.source.layout,
            )
            (slice_, allocation), = self.device.upload_checkpoint_tensors_with_allocations(
                [(tensor.name, checkpoint_tensor)]
            )
        with tensor.runtime_write_scope():
            tensor.buffer = slice_
            tensor.descriptor_nbytes = slice_.nbytes
        self._model_allocations.append(allocation)

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
            (slice_, allocation), = self.device.upload_numpy_arrays_with_allocations([(tensor.name, array)])
            with tensor.runtime_write_scope():
                tensor.buffer = slice_
                tensor.descriptor_nbytes = slice_.nbytes
            self._request_allocations.append(allocation)
            return
        with tensor.runtime_write_scope():
            tensor.buffer = BufferSlice(allocation=allocation, offset=allocation.offset, nbytes=expected)
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
            data[field.offset : field.offset + field.size] = _pack_push_constant_value(field.dtype, value)
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
        compiler = shutil.which("glslangValidator")
        if compiler is None:
            raise RuntimeError("glslangValidator is required to compile inline ShaderVariant source")
        source_hash = hashlib.sha256(variant.source.encode("utf-8")).hexdigest()[:16]
        stem = f"{variant.name}.{source_hash}"
        glsl_path = self.artifact_dir / f"{stem}.comp"
        spv_path = self.artifact_dir / f"{stem}.spv"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if not spv_path.is_file():
            glsl_path.write_text(variant.source, encoding="utf-8")
            subprocess.run([compiler, "-V", str(glsl_path), "-o", str(spv_path)], check=True)
        return spv_path

    def _release_frame_allocations(self) -> None:
        while self._frame_allocations:
            tensor, allocation = self._frame_allocations.pop()
            if tensor.buffer is not None and tensor.buffer.allocation is allocation:
                with tensor.runtime_write_scope():
                    tensor.buffer = None
                    tensor.descriptor_nbytes = None
            allocation.close()


def _tensor_nbytes(spec: TensorSpec) -> int:
    concrete_shape: list[int] = []
    for dim in spec.shape:
        if not isinstance(dim, int):
            raise ValueError(f"Expected concrete tensor shape, got {spec.shape}")
        concrete_shape.append(dim)
    return concrete_nbytes(dtype=spec.dtype, shape=tuple(concrete_shape))


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


def _record_descriptor_view(index: int, field: TensorFieldSpec, tensor: LogicalTensor) -> tuple[str, int, int, int]:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} is not materialized")
    return (
        field.name,
        index,
        tensor.buffer.offset,
        tensor.descriptor_nbytes or 0,
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


def _call_reference_model(
    reference_model: object,
    *,
    inputs: Mapping[LogicalTensor, object],
    scope: FrameScope,
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
        result = reference_model(dict(inputs), scope)
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
