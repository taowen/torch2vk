"""REQUEST_STATE tensor upload, growth, and release helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from vulkan import (
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
)

from torch2vk.runtime.host_array import prepare_host_array
from torch2vk.runtime.logical import LogicalTensor, MemoryClass, TensorSemantic
from torch2vk.vulkan.allocation import BufferAllocation, BufferSlice
from torch2vk.vulkan.device import VulkanDevice
from torch2vk.vulkan.types import concrete_shape, dtype_nbytes, tensor_nbytes

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


def initialize_request_state(
    rt: RuntimeSession,
    states: Mapping[LogicalTensor, object],
) -> None:
    """Upload initial values for REQUEST_STATE tensors such as KV caches."""
    rt._require_open()
    for tensor, value in states.items():
        if not isinstance(tensor, LogicalTensor):
            raise TypeError(
                f"initialize_request_state key must be LogicalTensor, got {type(tensor).__name__}"
            )
        tensor.validate_declaration()
        if tensor.memory is not MemoryClass.REQUEST_STATE:
            raise ValueError(f"{tensor.name} is not REQUEST_STATE memory")
        array = prepare_host_array(tensor, value, context="request state")
        expected = tensor_nbytes(tensor.spec)
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
            ((slice_, allocation),) = rt.device.upload_numpy_arrays_with_allocations(
                [(tensor.name, array)]
            )
            with tensor.runtime_write_scope():
                tensor.buffer = slice_
                tensor.descriptor_nbytes = expected
            rt._request_allocations.append(allocation)
            continue
        ((source, allocation),) = rt.device.upload_numpy_arrays_with_allocations(
            [(tensor.name, array)]
        )
        try:
            rt.device.copy_buffer(
                source.allocation.buffer,
                tensor.buffer.allocation.buffer,
                expected,
                src_offset=source.offset,
                dst_offset=tensor.buffer.offset,
            )
        finally:
            allocation.close()


def read_request_state(rt: RuntimeSession, tensor: LogicalTensor) -> np.ndarray:
    """Read back a materialized REQUEST_STATE tensor."""
    rt._require_open()
    tensor.validate_declaration()
    if tensor.memory is not MemoryClass.REQUEST_STATE:
        raise ValueError(f"{tensor.name} is not REQUEST_STATE memory")
    return rt.readback(tensor)


def grow_request_state(
    rt: RuntimeSession,
    tensor: LogicalTensor,
    new_shape: Sequence[int],
    *,
    growth: str = "geometric",
) -> None:
    """Grow a REQUEST_STATE tensor's logical shape and backing capacity."""
    rt._require_open()
    tensor.validate_declaration()
    if tensor.memory is not MemoryClass.REQUEST_STATE:
        raise ValueError(f"{tensor.name} is not REQUEST_STATE memory")
    if tensor.semantic not in {TensorSemantic.KV_CACHE, TensorSemantic.TOKEN}:
        raise ValueError(
            f"{tensor.name} request-state growth is only supported for TOKEN or KV_CACHE tensors"
        )
    if growth != "geometric":
        raise ValueError(f"Unsupported request-state growth strategy: {growth!r}")

    old_shape = concrete_shape(tensor.spec)
    resolved_new_shape = tuple(int(dim) for dim in new_shape)
    _validate_request_state_growth_shape(
        tensor=tensor,
        old_shape=old_shape,
        new_shape=resolved_new_shape,
    )
    if resolved_new_shape == old_shape:
        return

    old_logical_nbytes = tensor_nbytes(tensor.spec)
    new_spec = tensor.spec.with_shape(*resolved_new_shape)
    new_logical_nbytes = tensor_nbytes(new_spec)
    requires_relayout = _request_state_growth_requires_relayout(
        tensor=tensor,
        old_shape=old_shape,
        new_shape=resolved_new_shape,
    )
    if new_logical_nbytes == 0:
        with tensor.runtime_write_scope():
            tensor.spec = new_spec
            tensor.descriptor_nbytes = 0
            tensor.version += 1
        return
    buffer = tensor.buffer
    if buffer is not None and buffer.nbytes >= new_logical_nbytes and not requires_relayout:
        with tensor.runtime_write_scope():
            tensor.spec = new_spec
            tensor.descriptor_nbytes = new_logical_nbytes
            tensor.version += 1
        return

    new_capacity_nbytes = _grown_request_state_capacity_nbytes(
        old_capacity_nbytes=0 if buffer is None else buffer.nbytes,
        required_nbytes=new_logical_nbytes,
    )
    if requires_relayout and buffer is not None:
        new_capacity_nbytes = max(new_capacity_nbytes, buffer.nbytes)
    new_allocation = rt.device.memory_manager.allocate_device_local_buffer(
        new_capacity_nbytes,
        usage_flags=(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        ),
    )
    old_allocation = None if buffer is None else buffer.allocation
    try:
        if buffer is not None and old_logical_nbytes > 0:
            _copy_grown_request_state(
                rt,
                tensor=tensor,
                source=buffer,
                destination=new_allocation,
                old_shape=old_shape,
                new_shape=resolved_new_shape,
                old_logical_nbytes=old_logical_nbytes,
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
        rt._request_allocations.append(new_allocation)
    except Exception:
        new_allocation.close()
        raise
    if old_allocation is not None:
        rt._release_request_allocation(old_allocation)


def release_request_state(
    rt: RuntimeSession,
    tensors: Sequence[LogicalTensor] | None = None,
) -> None:
    """Release selected request-state buffers, or all request allocations when omitted."""
    rt._require_open()
    if tensors is None:
        for allocation in reversed(rt._request_allocations):
            allocation.close()
        rt._request_allocations.clear()
        return
    requested = set(tensors)
    retained: list[BufferAllocation] = []
    released: list[BufferAllocation] = []
    for allocation in rt._request_allocations:
        if any(
            tensor.buffer is not None and tensor.buffer.allocation is allocation
            for tensor in requested
        ):
            allocation.close()
            released.append(allocation)
        else:
            retained.append(allocation)
    rt._request_allocations = retained
    for tensor in requested:
        if tensor.buffer is not None and any(
            tensor.buffer.allocation is allocation for allocation in released
        ):
            with tensor.runtime_write_scope():
                tensor.buffer = None
                tensor.descriptor_nbytes = None
                tensor.writer = None


def _copy_grown_request_state(
    rt: RuntimeSession,
    *,
    tensor: LogicalTensor,
    source: BufferSlice,
    destination: BufferAllocation,
    old_shape: tuple[int, ...],
    new_shape: tuple[int, ...],
    old_logical_nbytes: int,
) -> None:
    descriptor_nbytes = (
        old_logical_nbytes if tensor.descriptor_nbytes is None else tensor.descriptor_nbytes
    )
    readable_nbytes = min(old_logical_nbytes, descriptor_nbytes, source.nbytes)
    if tensor.semantic is TensorSemantic.KV_CACHE:
        if readable_nbytes < old_logical_nbytes:
            raise RuntimeError(
                f"{tensor.name} cannot preserve partially materialized KV cache "
                f"({readable_nbytes} of {old_logical_nbytes} bytes)"
            )
        _copy_grown_kv_cache(
            device=rt.device,
            source=source,
            destination=destination,
            dtype=tensor.spec.dtype,
            old_shape=old_shape,
            new_shape=new_shape,
        )
        return
    if readable_nbytes > 0:
        rt.device.copy_buffer(
            source.allocation.buffer,
            destination.buffer,
            readable_nbytes,
            src_offset=source.offset,
            dst_offset=destination.offset,
        )


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


def _request_state_growth_requires_relayout(
    *,
    tensor: LogicalTensor,
    old_shape: tuple[int, ...],
    new_shape: tuple[int, ...],
) -> bool:
    return tensor.semantic is TensorSemantic.KV_CACHE and old_shape != new_shape


def _copy_grown_kv_cache(
    *,
    device: VulkanDevice,
    source: BufferSlice,
    destination: BufferAllocation,
    dtype: str,
    old_shape: tuple[int, ...],
    new_shape: tuple[int, ...],
) -> None:
    batch, heads, old_sequence, head_dim = old_shape
    new_batch, new_heads, new_sequence, new_head_dim = new_shape
    if (batch, heads, head_dim) != (new_batch, new_heads, new_head_dim):
        raise ValueError(
            "KV cache relayout only supports growing sequence dimension; "
            f"old_shape={old_shape}, new_shape={new_shape}"
        )
    element_nbytes = dtype_nbytes(dtype)
    segment_nbytes = old_sequence * head_dim * element_nbytes
    if segment_nbytes == 0:
        return
    source_stride_nbytes = segment_nbytes
    destination_stride_nbytes = new_sequence * head_dim * element_nbytes
    transfers = [
        (
            source.offset + segment * source_stride_nbytes,
            destination.buffer,
            destination.offset + segment * destination_stride_nbytes,
            segment_nbytes,
        )
        for segment in range(batch * heads)
    ]
    device.copy_buffer_transfers(source.allocation.buffer, transfers)


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
