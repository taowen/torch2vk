"""LogicalTensor materialization helpers for RuntimeSession."""

from __future__ import annotations

import struct
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from vulkan import (
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
)

from torch2vk.checkpoints.checkpoint_tensor import CheckpointTensor
from torch2vk.runtime.host_array import prepare_host_array
from torch2vk.runtime.logical import LogicalTensor, MemoryClass, TensorLifetime, TensorRole
from torch2vk.runtime.shader import (
    DTypeReference,
    DispatchTensorSnapshot,
    IOKind,
    ParamsBufferSpec,
    PushConstantInput,
    PushConstantSpec,
    PushConstantType,
    TensorFieldSpec,
    eval_expr,
)
from torch2vk.vulkan.allocation import BufferAllocation, BufferSlice
from torch2vk.vulkan.compute_pipeline import DescriptorBufferBinding
from torch2vk.vulkan.types import bind_tensor_layout_symbols, concrete_shape, tensor_nbytes

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession

_EMPTY_DESCRIPTOR_NBYTES = 4


def bind_shape_symbols(
    rt: RuntimeSession,
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


def materialize_read(rt: RuntimeSession, tensor: LogicalTensor) -> None:
    if tensor.alias_source is not None:
        source = tensor.alias_source
        materialize_read(rt, source)
        if source.buffer is None:
            raise RuntimeError(f"{tensor.name} alias source {source.name} is not materialized")
        with tensor.runtime_write_scope():
            tensor.buffer = source.buffer
            tensor.descriptor_nbytes = source.descriptor_nbytes
            tensor.version = source.version
            tensor.writer = source.writer
        return
    if tensor.buffer is not None:
        return
    if tensor.role is TensorRole.WEIGHT:
        materialize_weight(rt, tensor)
        return
    if tensor.memory is MemoryClass.SESSION_TENSOR:
        raise RuntimeError(f"{tensor.name} requires initialize_session_tensors(...) before use")
    if tensor.role is TensorRole.INPUT:
        materialize_input(rt, tensor)
        return
    raise RuntimeError(f"{tensor.name} cannot be read before it is materialized or written")


def materialize_write(rt: RuntimeSession, tensor: LogicalTensor, *, io_kind: IOKind) -> None:
    if tensor.alias_source is not None:
        raise RuntimeError(
            f"{tensor.name} is an alias of {tensor.alias_source.name} and cannot be written"
        )
    if io_kind is IOKind.INOUT and tensor.buffer is not None:
        return
    if (
        tensor.memory is MemoryClass.REQUEST_STATE
        and tensor.buffer is not None
        and not tensor.buffer.allocation.released
    ):
        return
    size = tensor_nbytes(tensor.spec)
    if size == 0:
        allocation = rt.device.memory_manager.allocate_device_local_buffer(
            _EMPTY_DESCRIPTOR_NBYTES,
            usage_flags=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        )
        with tensor.runtime_write_scope():
            tensor.buffer = BufferSlice(
                allocation=allocation,
                offset=allocation.offset,
                nbytes=_EMPTY_DESCRIPTOR_NBYTES,
            )
            tensor.descriptor_nbytes = _EMPTY_DESCRIPTOR_NBYTES
            tensor.alias_source = None
        if tensor.memory is MemoryClass.REQUEST_STATE:
            rt._request_allocations.append(allocation)
            rt._request_tensors.add(tensor)
        if tensor.lifetime in {TensorLifetime.FRAME, TensorLifetime.OP}:
            rt._frame_allocations.append((tensor, allocation))
        return
    materialized_slice: BufferSlice | None = None
    if tensor.memory is MemoryClass.HOST_OUTPUT:
        allocation = rt.device.allocate_host_visible_allocation(size)
    elif tensor.memory in {MemoryClass.FRAME_WORKSPACE, MemoryClass.OP_SCRATCH}:
        materialized_slice, allocation = rt.device.memory_manager.temporary_tensor_pool.acquire(
            tensor.spec,
            label=tensor.name,
        )
    elif tensor.memory is MemoryClass.REQUEST_STATE:
        allocation = rt.device.memory_manager.allocate_device_local_buffer(
            size,
            usage_flags=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        )
        rt._request_allocations.append(allocation)
    else:
        raise ValueError(
            f"{tensor.name} cannot be materialized for write with memory={tensor.memory}"
        )
    with tensor.runtime_write_scope():
        tensor.buffer = (
            materialized_slice
            if materialized_slice is not None
            else BufferSlice(allocation=allocation, offset=allocation.offset, nbytes=size)
        )
        tensor.descriptor_nbytes = size
        tensor.alias_source = None
    if tensor.memory is MemoryClass.REQUEST_STATE:
        rt._request_tensors.add(tensor)
    if tensor.lifetime in {TensorLifetime.FRAME, TensorLifetime.OP}:
        rt._frame_allocations.append((tensor, allocation))


def materialize_weight(rt: RuntimeSession, tensor: LogicalTensor) -> None:
    if tensor.buffer is not None:
        return
    tensor.validate_declaration()
    checkpoint = resolve_weight_checkpoint(rt, tensor)
    tensor_key = tensor.checkpoint_key
    if not tensor_key:
        raise RuntimeError(f"{tensor.name} is a weight tensor but checkpoint_key is not set")
    cached = rt._cached_weight(
        tensor_key=tensor_key,
        checkpoint=checkpoint,
        spec=tensor.spec,
        layout=tensor.layout,
    )
    if cached is not None:
        with tensor.runtime_write_scope():
            tensor.buffer = cached.buffer
            tensor.descriptor_nbytes = cached.descriptor_nbytes
            tensor.alias_source = None
        return
    checkpoint_tensor = CheckpointTensor.open(
        storage=rt._checkpoint_storage(checkpoint),
        tensor_key=tensor_key,
        dtype=tensor.spec.dtype,
        shape=tensor.concrete_shape,
        layout=tensor.layout,
    )
    ((slice_, allocation),) = rt.device.upload_checkpoint_tensors_with_allocations(
        [(tensor.name, checkpoint_tensor)]
    )
    with tensor.runtime_write_scope():
        tensor.buffer = slice_
        tensor.descriptor_nbytes = slice_.nbytes
    rt._cache_weight(
        tensor_key=tensor_key,
        checkpoint=checkpoint,
        spec=tensor.spec,
        layout=tensor.layout,
        buffer=slice_,
        descriptor_nbytes=slice_.nbytes,
        allocation=allocation,
    )


def resolve_weight_checkpoint(rt: RuntimeSession, tensor: LogicalTensor) -> Path:
    if rt.model_dir is None:
        raise RuntimeError(
            f"{tensor.name} is a weight tensor but RuntimeSession has no model_dir to resolve checkpoint"
        )
    if tensor.checkpoint:
        explicit = (rt.model_dir / tensor.checkpoint).resolve()
        if explicit.is_file():
            return explicit
        raise RuntimeError(
            f"{tensor.name} requested checkpoint {tensor.checkpoint!r} but file does not exist: {explicit}"
        )
    primary = rt.model_dir / "model.safetensors"
    if primary.is_file():
        return primary
    index = rt.model_dir / "model.safetensors.index.json"
    if index.is_file():
        return index
    gguf = rt.model_dir / "model.gguf"
    if gguf.is_file():
        return gguf
    candidates = (
        sorted(rt.model_dir.glob("*.safetensors"))
        + sorted(rt.model_dir.glob("*.safetensors.index.json"))
        + sorted(rt.model_dir.glob("*.gguf"))
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            f"{tensor.name} could not find a safetensors checkpoint in {rt.model_dir}"
        )
    raise RuntimeError(
        f"{tensor.name} found multiple safetensors checkpoints in {rt.model_dir}; "
        "use model.safetensors or model.safetensors.index.json as the canonical checkpoint"
    )


def materialize_input(rt: RuntimeSession, tensor: LogicalTensor) -> None:
    if tensor not in rt._inputs:
        raise RuntimeError(f"{tensor.name} requires missing input")
    value = rt._inputs[tensor]
    array = prepare_host_array(tensor, value, context="input")
    expected = tensor_nbytes(tensor.spec)
    if array.nbytes != expected:
        raise ValueError(f"{tensor.name} input has {array.nbytes} bytes, expected {expected}")
    if tensor.memory is MemoryClass.HOST_INPUT:
        allocation = rt.device.allocate_host_visible_allocation(expected)
        allocation.buffer.write_bytes_at(allocation.offset, memoryview(array).cast("B"))
        rt.device.memory_manager.host_upload_ring.flush(allocation=allocation)
    else:
        ((slice_, allocation),) = rt.device.upload_numpy_arrays_with_allocations(
            [(tensor.name, array)]
        )
        with tensor.runtime_write_scope():
            tensor.buffer = slice_
            tensor.descriptor_nbytes = slice_.nbytes
            tensor.alias_source = None
        rt._request_allocations.append(allocation)
        rt._request_tensors.add(tensor)
        return
    with tensor.runtime_write_scope():
        tensor.buffer = BufferSlice(
            allocation=allocation, offset=allocation.offset, nbytes=expected
        )
        tensor.descriptor_nbytes = expected
        tensor.alias_source = None
    if tensor.lifetime in {TensorLifetime.FRAME, TensorLifetime.OP}:
        rt._frame_allocations.append((tensor, allocation))


def pack_push_constants(
    rt: RuntimeSession,
    spec: PushConstantSpec | None,
    *,
    tensors: Mapping[str, LogicalTensor],
    symbols: Mapping[str, int],
    push_constant_inputs: Mapping[str, object] | None = None,
) -> tuple[bytes | None, dict[str, int | float]]:
    if spec is None:
        return None, {}
    inputs = {} if push_constant_inputs is None else push_constant_inputs
    data = bytearray(spec.size)
    values: dict[str, int | float] = {}
    for field in spec.fields:
        raw = field.value
        if isinstance(raw, PushConstantInput):
            value = _push_constant_input_value(field.name, raw.name, inputs)
        elif callable(raw):
            value = raw(tensors, symbols)
        elif isinstance(raw, str):
            value = symbols[raw]
        elif isinstance(raw, int | float):
            value = raw
        else:
            value = eval_expr(raw, symbols)
        values[field.name] = value
        data[field.offset : field.offset + field.size] = pack_push_constant_value(
            field.dtype, value
        )
    return bytes(data), values


def _push_constant_input_value(
    field_name: str,
    input_name: str,
    inputs: Mapping[str, object],
) -> int | float:
    if input_name not in inputs:
        raise ValueError(f"Push constant {field_name!r} is missing input {input_name!r}")
    value = inputs[input_name]
    if isinstance(value, bool):
        raise TypeError(f"Push constant {field_name!r} cannot be bool")
    if isinstance(value, int | float):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if isinstance(scalar, bool):
            raise TypeError(f"Push constant {field_name!r} cannot be bool")
        if isinstance(scalar, int | float):
            return scalar
    raise TypeError(
        f"Push constant {field_name!r} expects a scalar int or float, got {type(value).__name__}"
    )


def materialize_params_buffer(
    rt: RuntimeSession,
    spec: ParamsBufferSpec,
    *,
    tensors: Mapping[str, LogicalTensor],
    symbols: Mapping[str, int],
    push_constant_inputs: Mapping[str, object] | None = None,
) -> BufferAllocation:
    allocation = rt.device.allocate_host_visible_allocation(spec.size)
    inputs = {} if push_constant_inputs is None else push_constant_inputs
    data = bytearray(spec.size)
    for field in spec.fields:
        raw = field.value
        if isinstance(raw, PushConstantInput):
            value = _push_constant_input_value(field.name, raw.name, inputs)
        elif callable(raw):
            value = raw(tensors, symbols)
        elif isinstance(raw, str):
            value = symbols[raw]
        elif isinstance(raw, int | float):
            value = raw
        else:
            value = eval_expr(raw, symbols)
        data[field.offset : field.offset + field.size] = pack_push_constant_value(
            field.dtype, value
        )
    allocation.buffer.write_bytes_at(allocation.offset, bytes(data))
    return allocation


def release_frame_allocations(rt: RuntimeSession) -> None:
    while rt._frame_allocations:
        tensor, allocation = rt._frame_allocations.pop()
        if tensor.buffer is not None and tensor.buffer.allocation is allocation:
            with tensor.runtime_write_scope():
                tensor.buffer = None
                tensor.descriptor_nbytes = None
        if tensor.memory in {MemoryClass.FRAME_WORKSPACE, MemoryClass.OP_SCRATCH}:
            rt.device.memory_manager.temporary_tensor_pool.recycle(
                spec=tensor.spec,
                allocation=allocation,
            )
        else:
            allocation.close()


def release_request_allocation(rt: RuntimeSession, allocation: BufferAllocation) -> None:
    rt._request_allocations = [
        request_allocation
        for request_allocation in rt._request_allocations
        if request_allocation is not allocation
    ]
    allocation.close()
    for tensor in tuple(rt._request_tensors):
        if tensor.buffer is not None and tensor.buffer.allocation is allocation:
            with tensor.runtime_write_scope():
                tensor.buffer = None
                tensor.descriptor_nbytes = None
                tensor.writer = None
                tensor.alias_source = None
            rt._request_tensors.discard(tensor)


def descriptor_view_for_field(
    field: TensorFieldSpec,
    tensor: LogicalTensor,
) -> tuple[TensorFieldSpec, DescriptorBufferBinding]:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} is not materialized")
    return field, DescriptorBufferBinding.from_slice(
        tensor.buffer,
        descriptor_nbytes=tensor.descriptor_nbytes,
    )


def record_descriptor_view(
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


def record_tensor_snapshot(field: TensorFieldSpec, tensor: LogicalTensor) -> DispatchTensorSnapshot:
    if tensor.buffer is None:
        raise RuntimeError(f"{tensor.name} is not materialized")
    return DispatchTensorSnapshot(
        field=field.name,
        tensor=tensor.name,
        shape=concrete_shape(tensor.spec),
        dtype=tensor.spec.dtype,
        descriptor_offset=tensor.buffer.offset,
        descriptor_nbytes=tensor.descriptor_nbytes or 0,
        version=tensor.version,
    )


def pack_push_constant_value(dtype: PushConstantType, value: int | float) -> bytes:
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
