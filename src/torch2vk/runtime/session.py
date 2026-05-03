"""RuntimeSession materialization, dispatch, and lifecycle orchestration."""

from __future__ import annotations

import hashlib
import shutil
import struct
import subprocess
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from vulkan import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT

from torch2vk.checkpoints.checkpoint_tensor import CheckpointTensor
from torch2vk.checkpoints.safetensors import open_safetensors_mmap
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
    TensorFieldSpec,
    eval_expr,
)
from torch2vk.vulkan.allocation import BufferAllocation, BufferSlice
from torch2vk.vulkan.compute_pipeline import ComputePipeline, DescriptorBufferBinding
from torch2vk.vulkan.device import VulkanDevice
from torch2vk.vulkan.types import TensorSpec, concrete_nbytes


class RuntimeSession:
    """The single runtime owner for LogicalTensor materialization and shader dispatch."""

    def __init__(self, *, device_index: int = 0, artifact_dir: str | Path | None = None) -> None:
        self.device = VulkanDevice(physical_device_index=device_index)
        self.artifact_dir = Path(".cache/torch2vk/generated" if artifact_dir is None else artifact_dir)
        self.model_dir: Path | None = None
        self._inputs: dict[LogicalTensor, object] = {}
        self._frame_stack: list[FrameContext] = []
        self._dispatch_records: list[DispatchRecord] = []
        self._pipeline_cache: dict[tuple[Any, ...], ComputePipeline] = {}
        self._model_allocations: list[BufferAllocation] = []
        self._request_allocations: list[BufferAllocation] = []
        self._frame_allocations: list[tuple[LogicalTensor, BufferAllocation]] = []
        self._closed = False

    @classmethod
    def open(cls, *, device_index: int = 0, artifact_dir: str | Path | None = None) -> "RuntimeSession":
        return cls(device_index=device_index, artifact_dir=artifact_dir)

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

    def register_model(self, tensors: Iterable[LogicalTensor], *, model_dir: str | Path | None = None) -> None:
        if model_dir is not None:
            self.model_dir = Path(model_dir).expanduser().resolve()
        names: set[str] = set()
        for tensor in tensors:
            tensor.validate_declaration()
            if tensor.name in names:
                raise ValueError(f"Duplicate LogicalTensor name {tensor.name}")
            names.add(tensor.name)

    def register_inputs(self, inputs: Mapping[LogicalTensor, object]) -> None:
        for tensor, value in inputs.items():
            if not isinstance(tensor, LogicalTensor):
                raise TypeError(f"register_inputs key must be LogicalTensor, got {type(tensor).__name__}")
            if tensor.role is not TensorRole.INPUT:
                raise ValueError(f"{tensor.name} is not an input tensor")
            self._inputs[tensor] = value

    @contextmanager
    def frame(
        self,
        name: str,
        *,
        scope: Mapping[str, str | int] | None = None,
        pytorch_model: object | None = None,
        reference_model: object | None = None,
    ):
        if not name:
            raise ValueError("frame name must be non-empty")
        context = FrameContext(
            scope=FrameScope(frame=name, values={} if scope is None else dict(scope)),
            start_dispatch_index=len(self._dispatch_records),
            pytorch_model=pytorch_model,
            reference_model=reference_model,
        )
        self._frame_stack.append(context)
        try:
            yield context
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
            tensors[name] = argument

        symbols = self._bind_shape_symbols(contract.fields, tensors)
        for field in contract.input_fields:
            self._materialize_read(tensors[field.name])
        for field in contract.output_fields:
            self._materialize_write(tensors[field.name])

        descriptor_bindings = tuple(
            _descriptor_binding_for_field(field, tensors[field.name]) for field in contract.fields
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
            buffers=[binding for _, binding in descriptor_bindings],
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
            reads=tuple((field.name, tensors[field.name].name) for field in contract.input_fields),
            writes=tuple((field.name, tensors[field.name].name) for field in contract.output_fields),
            symbols=tuple(sorted(symbols.items())),
            dispatch_size=dispatch_size,
            descriptor_bindings=tuple(_record_descriptor_binding(field, tensors[field.name]) for field in contract.fields),
            push_constant_values=tuple(sorted(push_values.items())),
        )
        self._dispatch_records.append(record)
        for field in contract.output_fields:
            tensor = tensors[field.name]
            tensor.version += 1
            tensor.writer = DispatchWriter(
                frame=frame.scope.frame,
                shader=variant.name,
                dispatch_index=index,
            )

    def readback(self, tensor: LogicalTensor) -> np.ndarray:
        self._require_open()
        if tensor.buffer is None:
            raise RuntimeError(f"{tensor.name} is not materialized")
        return self.device.readback_tensor(spec=tensor.spec, slice=tensor.buffer, layout=tensor.layout)

    def debug_materialization(self, tensor: LogicalTensor) -> BufferSlice | None:
        return tensor.buffer

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
        return symbols

    def _materialize_read(self, tensor: LogicalTensor) -> None:
        if tensor.buffer is not None:
            return
        if tensor.source is not None:
            self._materialize_weight(tensor)
            return
        if tensor.role is TensorRole.INPUT:
            self._materialize_input(tensor)
            return
        raise RuntimeError(f"{tensor.name} cannot be read before it is materialized or written")

    def _materialize_write(self, tensor: LogicalTensor) -> None:
        if tensor.buffer is not None and tensor.lifetime is not TensorLifetime.OP:
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
            allocation.buffer.write_bytes_at(allocation.offset, memoryview(array))
            self.device.memory_manager.host_upload_ring.flush(allocation=allocation)
        else:
            (slice_, allocation), = self.device.upload_numpy_arrays_with_allocations([(tensor.name, array)])
            tensor.buffer = slice_
            tensor.descriptor_nbytes = slice_.nbytes
            self._request_allocations.append(allocation)
            return
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
        descriptor_bindings = tuple(field.binding for field in variant.contract.fields)
        key = (
            str(spv_path),
            descriptor_bindings,
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
            descriptor_bindings=descriptor_bindings,
            storage_buffer_count=len(descriptor_bindings),
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


def _descriptor_binding_for_field(
    field: TensorFieldSpec,
    tensor: LogicalTensor,
) -> tuple[TensorFieldSpec, DescriptorBufferBinding]:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} is not materialized")
    return field, DescriptorBufferBinding.from_slice(
        tensor.buffer,
        descriptor_nbytes=tensor.descriptor_nbytes,
    )


def _record_descriptor_binding(field: TensorFieldSpec, tensor: LogicalTensor) -> tuple[str, int, int, int]:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} is not materialized")
    return (
        field.name,
        field.binding,
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
