"""Owning Vulkan device with explicit allocation and transfer helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from _cffi_backend import _CDataBase
from dataclasses import dataclass
from types import TracebackType

import numpy as np
from vulkan import (
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    VK_NULL_HANDLE,
    VkBufferCopy,
    VkCommandBufferAllocateInfo,
    VkCommandBufferBeginInfo,
    VkCommandPoolCreateInfo,
    vkAllocateCommandBuffers,
    vkBeginCommandBuffer,
    vkCmdCopyBuffer,
    vkCmdFillBuffer,
    vkCreateCommandPool,
    vkDestroyCommandPool,
    vkDestroyDevice,
    vkDestroyInstance,
    vkDeviceWaitIdle,
    vkEndCommandBuffer,
    vkEnumeratePhysicalDevices,
    vkFreeCommandBuffers,
    vkGetDeviceQueue,
    vkGetPhysicalDeviceMemoryProperties,
    vkGetPhysicalDeviceProperties,
    vkDestroyFence,
)

from torch2vk.checkpoints.checkpoint_tensor import CheckpointTensor
from torch2vk.vulkan.types import CONTIGUOUS_LAYOUT, Residency, TensorLayout, TensorSpec

from .abi import VkPhysicalDeviceMemoryProperties, physical_device_memory_properties, physical_device_properties
from .allocation import BufferAllocation, BufferOwner, BufferSlice
from .bootstrap import create_device, create_instance, enumerate_device_extension_names, find_compute_queue_family
from .capabilities import (
    CooperativeMatrixSupport,
    DeviceFeatureSupport,
    IntegerDotProductSupport,
    SubgroupSizeControlSupport,
    query_cooperative_matrix_support,
    query_device_feature_support,
    query_integer_dot_product_support,
    query_subgroup_size_control_support,
    require_cooperative_matrix_requirements,
    require_shader_execution_requirements,
    require_subgroup_requirements,
    supports_shader_execution_requirements,
)
from .debug_utils import DebugUtils, create_debug_utils
from .memory_allocator import set_device_memory_priority, tensor_nbytes
from .memory_allocator import VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_VALUE
from .memory_manager import MemoryAllocationStats, MemoryManager
from .queue_submission import (
    submit_and_wait,
    submit_and_wait_with_fence,
    submit_one_shot_and_wait,
    submit_with_existing_fence,
    submit_with_new_fence,
    wait_for_fence,
)
from .shader_execution_requirements import (
    CooperativeMatrixRequirements,
    ShaderExecutionRequirements,
    SubgroupRequirements,
)


_SPEC_DTYPE_TO_NUMPY: dict[str, np.dtype] = {
    "bool": np.dtype("uint32"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "float16": np.dtype("float16"),
    "int8": np.dtype("int8"),
    "uint8": np.dtype("uint8"),
    "int16": np.dtype("int16"),
    "uint16": np.dtype("uint16"),
    "int32": np.dtype("int32"),
    "uint32": np.dtype("uint32"),
    "int64": np.dtype("int64"),
    "bfloat16": np.dtype("uint16"),
}
_NUMPY_DTYPE_TO_SPEC: dict[np.dtype, str] = {
    numpy_dtype: spec_dtype
    for spec_dtype, numpy_dtype in _SPEC_DTYPE_TO_NUMPY.items()
    if spec_dtype != "bfloat16"
}
_UPLOAD_STAGING_CHUNK_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_PENDING_SUBMITS = 64


def _configured_max_pending_submits() -> int:
    raw = os.environ.get("TORCH2VK_MAX_PENDING_SUBMITS")
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_PENDING_SUBMITS
    value = int(raw)
    if value <= 0:
        raise ValueError(f"TORCH2VK_MAX_PENDING_SUBMITS must be positive, got {raw!r}")
    return value


@dataclass(slots=True)
class _PendingCommandBufferSubmit:
    submit_id: int
    command_buffer: object
    fence: object
    cleanup: Callable[[], None]
    release_command_buffer: bool
    destroy_fence: bool


def _slice_for_spec(*, spec: TensorSpec, allocation: BufferAllocation) -> BufferSlice:
    return BufferSlice(allocation=allocation, offset=allocation.offset, nbytes=tensor_nbytes(spec))


def _numpy_dtype_for_spec(dtype: str) -> np.dtype:
    try:
        return _SPEC_DTYPE_TO_NUMPY[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype for numpy conversion: {dtype}") from exc


def _spec_dtype_for_numpy(dtype: np.dtype) -> str:
    normalized = np.dtype(dtype)
    try:
        return _NUMPY_DTYPE_TO_SPEC[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported numpy tensor dtype: {normalized}") from exc


def _decode_spec_array(*, spec: TensorSpec, array: np.ndarray) -> np.ndarray:
    if spec.dtype == "bool":
        return np.asarray(array != 0, dtype=np.bool_)
    return array


def _numpy_spec(array: np.ndarray) -> TensorSpec:
    return TensorSpec(
        dtype=_spec_dtype_for_numpy(array.dtype),
        shape=tuple(int(dim) for dim in array.shape),
        residency=Residency.DEVICE,
    )


class VulkanDevice:
    """Owns one Vulkan instance/device/queue and exposes concrete device operations."""

    def __init__(self, *, physical_device_index: int = 0) -> None:
        self._closed = False
        self.instance = create_instance()
        self.physical_device = vkEnumeratePhysicalDevices(self.instance)[physical_device_index]
        self.queue_family_index = find_compute_queue_family(self.physical_device)
        self.available_device_extensions = enumerate_device_extension_names(self.physical_device)
        self.device_feature_support: DeviceFeatureSupport = query_device_feature_support(self.physical_device)
        self.subgroup_size_control_support: SubgroupSizeControlSupport = query_subgroup_size_control_support(
            self.physical_device
        )
        self.integer_dot_product_support: IntegerDotProductSupport = query_integer_dot_product_support(
            self.physical_device
        )
        self.cooperative_matrix_support: tuple[CooperativeMatrixSupport, ...] = query_cooperative_matrix_support(
            instance=self.instance,
            physical_device=self.physical_device,
            available_extensions=self.available_device_extensions,
        )
        self.device = create_device(
            self.physical_device,
            self.queue_family_index,
            available_extensions=self.available_device_extensions,
            feature_support=self.device_feature_support,
        )
        self.debug_utils: DebugUtils = create_debug_utils(instance=self.instance, device=self.device)
        self.queue = vkGetDeviceQueue(self.device, self.queue_family_index, 0)
        self.memory_properties: VkPhysicalDeviceMemoryProperties = physical_device_memory_properties(
            vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        )
        self.device_properties = physical_device_properties(vkGetPhysicalDeviceProperties(self.physical_device))
        self.timestamp_period_ns = float(self.device_properties.timestamp_period)
        self.memory_manager = MemoryManager(
            device_handle=self.device,
            memory_properties=self.memory_properties,
            require_device_open=self.require_open,
            is_device_closed=lambda: self.closed,
        )
        self.command_pool = vkCreateCommandPool(
            self.device,
            VkCommandPoolCreateInfo(
                queueFamilyIndex=self.queue_family_index,
                flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            ),
            None,
        )
        self._pending_submits: list[_PendingCommandBufferSubmit] = []
        self._max_pending_submits = _configured_max_pending_submits()

    @property
    def closed(self) -> bool:
        return self._closed

    def supports_shader_execution_requirements(
        self,
        execution_requirements: ShaderExecutionRequirements | None,
    ) -> bool:
        return supports_shader_execution_requirements(
            feature_support=self.device_feature_support,
            subgroup_size_control_support=self.subgroup_size_control_support,
            cooperative_matrix_support=self.cooperative_matrix_support,
            execution_requirements=execution_requirements,
        )

    def require_shader_execution_requirements(
        self,
        execution_requirements: ShaderExecutionRequirements | None,
    ) -> None:
        require_shader_execution_requirements(
            feature_support=self.device_feature_support,
            subgroup_size_control_support=self.subgroup_size_control_support,
            cooperative_matrix_support=self.cooperative_matrix_support,
            execution_requirements=execution_requirements,
        )

    def require_subgroup_requirements(self, subgroup: SubgroupRequirements) -> None:
        require_subgroup_requirements(
            feature_support=self.device_feature_support,
            subgroup_size_control_support=self.subgroup_size_control_support,
            subgroup=subgroup,
        )

    def require_cooperative_matrix_requirements(self, cooperative_matrix: CooperativeMatrixRequirements) -> None:
        require_cooperative_matrix_requirements(
            cooperative_matrix_support=self.cooperative_matrix_support,
            cooperative_matrix=cooperative_matrix,
        )

    def require_open(self) -> None:
        if self.closed:
            raise RuntimeError("VulkanDevice is closed")

    @property
    def subgroup_size(self) -> int:
        return self.subgroup_size_control_support.subgroup_size

    def set_debug_name(self, *, object_type: int, handle: int | _CDataBase, name: str) -> None:
        self.require_open()
        self.debug_utils.set_object_name(
            device=self.device,
            object_type=object_type,
            handle=handle,
            name=name,
        )

    def begin_command_label(self, *, command_buffer: object, name: str) -> None:
        self.require_open()
        self.debug_utils.begin_command_label(command_buffer=command_buffer, name=name)

    def end_command_label(self, *, command_buffer: object) -> None:
        self.require_open()
        self.debug_utils.end_command_label(command_buffer=command_buffer)

    def wait_idle(self) -> None:
        self.require_open()
        self.wait_pending_submits()
        vkDeviceWaitIdle(self.device)

    def close(self) -> None:
        if self._closed:
            return
        if getattr(self, "device", VK_NULL_HANDLE):
            self.wait_pending_submits()
            self._closed = True
            vkDeviceWaitIdle(self.device)
            if hasattr(self, "memory_manager"):
                self.memory_manager.close()
            if getattr(self, "command_pool", VK_NULL_HANDLE):
                vkDestroyCommandPool(self.device, self.command_pool, None)
                self.command_pool = VK_NULL_HANDLE
            vkDestroyDevice(self.device, None)
            self.device = VK_NULL_HANDLE
        else:
            self._closed = True
        if getattr(self, "instance", VK_NULL_HANDLE):
            vkDestroyInstance(self.instance, None)
            self.instance = VK_NULL_HANDLE

    def __enter__(self) -> "VulkanDevice":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def allocate_tensor_allocation(
        self,
        spec: TensorSpec,
        *,
        layout: TensorLayout = CONTIGUOUS_LAYOUT,
        label: str | None = None,
        memory_priority: float | None = None,
    ) -> tuple[BufferSlice, BufferAllocation]:
        del layout, label
        self.require_open()
        if spec.residency is not Residency.DEVICE:
            raise ValueError(f"allocate_tensor_allocation requires device residency, got {spec.residency}")
        allocation = self.memory_manager.allocate_device_local_buffer(
            tensor_nbytes(spec),
            memory_priority=memory_priority,
        )
        return _slice_for_spec(spec=spec, allocation=allocation), allocation

    def set_memory_priority(self, allocation: BufferAllocation, priority: float) -> None:
        self.require_open()
        set_device_memory_priority(
            device_handle=self.device,
            memory=allocation.buffer.memory,
            priority=priority,
        )

    def allocate_host_visible_allocation(
        self,
        size: int,
        *,
        usage_flags: int = (
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_VALUE
        ),
    ) -> BufferAllocation:
        self.require_open()
        return self.memory_manager.allocate_host_visible_buffer(size, usage_flags=usage_flags)

    def allocate_host_visible_tensor_allocation(
        self,
        spec: TensorSpec,
        *,
        layout: TensorLayout = CONTIGUOUS_LAYOUT,
        label: str | None = None,
    ) -> tuple[BufferSlice, BufferAllocation]:
        del layout, label
        self.require_open()
        if spec.residency is not Residency.DEVICE:
            raise ValueError(
                f"allocate_host_visible_tensor_allocation requires device residency, got {spec.residency}"
            )
        allocation = self.memory_manager.allocate_host_visible_buffer(tensor_nbytes(spec))
        return _slice_for_spec(spec=spec, allocation=allocation), allocation

    def allocation_stats(self) -> MemoryAllocationStats:
        self.require_open()
        return self.memory_manager.allocation_stats()

    def allocation_epoch(self) -> int:
        self.require_open()
        return self.memory_manager.allocation_epoch()

    def reset_allocation_stats(self) -> None:
        self.require_open()
        self.memory_manager.reset_allocation_stats()

    def upload_numpy_arrays_with_allocations(
        self,
        tensors: tuple[tuple[str | None, np.ndarray], ...] | list[tuple[str | None, np.ndarray]],
    ) -> tuple[tuple[BufferSlice, BufferAllocation], ...]:
        uploads = tuple((label, np.ascontiguousarray(array)) for label, array in tensors)
        return self.upload_buffer_views_with_allocations(
            tuple(
                (label, _numpy_spec(array), memoryview(array).cast("B"))
                for label, array in uploads
            )
        )

    def upload_checkpoint_tensors_with_allocations(
        self,
        tensors: (
            tuple[tuple[str | None, CheckpointTensor], ...]
            | list[tuple[str | None, CheckpointTensor]]
        ),
    ) -> tuple[tuple[BufferSlice, BufferAllocation], ...]:
        return self.upload_buffer_views_with_allocations(
            tuple((label, tensor.spec, tensor.buffer_view()) for label, tensor in tensors),
            memory_priority=1.0,
        )

    def upload_buffer_views_with_allocations(
        self,
        tensors: (
            tuple[tuple[str | None, TensorSpec, bytes | bytearray | memoryview], ...]
            | list[tuple[str | None, TensorSpec, bytes | bytearray | memoryview]]
        ),
        *,
        memory_priority: float | None = None,
    ) -> tuple[tuple[BufferSlice, BufferAllocation], ...]:
        self.require_open()
        if not tensors:
            return ()

        uploads = tuple((label, spec, data) for label, spec, data in tensors)
        allocations = tuple(
            self.allocate_tensor_allocation(
                spec.with_residency(Residency.DEVICE),
                label=label,
                memory_priority=memory_priority,
            )
            for label, spec, _data in uploads
        )
        slices = tuple(slice for slice, _ in allocations)
        owned_allocations = tuple(allocation for _, allocation in allocations)

        for _, spec, data in uploads:
            expected_nbytes = tensor_nbytes(spec)
            if len(data) != expected_nbytes:
                raise ValueError(
                    f"Upload buffer for {spec} has {len(data)} bytes, expected {expected_nbytes}"
                )

        try:
            for slice, (_label, _spec, data) in zip(slices, uploads, strict=True):
                remaining = len(data)
                copied = 0
                while copied < remaining:
                    chunk_nbytes = min(_UPLOAD_STAGING_CHUNK_BYTES, remaining - copied)
                    staging_allocation = self.memory_manager.allocate_host_upload_buffer(
                        chunk_nbytes,
                        usage_flags=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    )
                    try:
                        staging = staging_allocation.buffer
                        staging.write_bytes_at(
                            staging_allocation.offset,
                            data[copied : copied + chunk_nbytes],
                        )
                        self.memory_manager.host_upload_ring.flush(allocation=staging_allocation)
                        self.copy_buffer_transfers(
                            staging,
                            (
                                (
                                    staging_allocation.offset,
                                    slice.allocation.buffer,
                                    slice.offset + copied,
                                    chunk_nbytes,
                                ),
                            ),
                        )
                    finally:
                        staging_allocation.close()
                    copied += chunk_nbytes
        except Exception:
            for allocation in reversed(owned_allocations):
                allocation.close()
            raise
        return allocations

    def readback_tensor(
        self,
        *,
        spec: TensorSpec,
        slice: BufferSlice,
        layout: TensorLayout = CONTIGUOUS_LAYOUT,
    ) -> np.ndarray:
        del layout
        self.require_open()
        expected_nbytes = tensor_nbytes(spec)
        if expected_nbytes == 0:
            return self.empty_tensor(spec=spec)
        data = self.readback_tensor_bytes(slice, size=expected_nbytes)
        dtype = _numpy_dtype_for_spec(spec.dtype)
        shape = tuple(int(dim) for dim in spec.shape)
        raw = np.frombuffer(data, dtype=dtype).copy().reshape(shape)
        return _decode_spec_array(spec=spec, array=raw)

    def empty_tensor(self, *, spec: TensorSpec) -> np.ndarray:
        dtype = np.dtype("bool") if spec.dtype == "bool" else _numpy_dtype_for_spec(spec.dtype)
        shape = tuple(int(dim) for dim in spec.shape)
        return np.empty(shape, dtype=dtype)

    def readback_tensor_bytes(
        self,
        slice: BufferSlice,
        *,
        byte_offset: int = 0,
        size: int | None = None,
    ) -> bytes:
        if byte_offset < 0:
            raise ValueError(f"Tensor readback byte_offset must be non-negative, got {byte_offset}")
        resolved_size = slice.nbytes - byte_offset if size is None else int(size)
        if resolved_size < 0:
            raise ValueError(f"Tensor readback size must be non-negative, got {resolved_size}")
        if byte_offset + resolved_size > slice.nbytes:
            raise ValueError(
                f"Tensor readback range [{byte_offset}, {byte_offset + resolved_size}) exceeds tensor size {slice.nbytes}"
            )
        return self.readback_buffer_range(
            slice.allocation.buffer,
            byte_offset=slice.offset + byte_offset,
            size=resolved_size,
        )

    def readback_buffer_range(
        self,
        buffer: BufferOwner,
        *,
        byte_offset: int = 0,
        size: int | None = None,
    ) -> bytes:
        self.require_open()
        self.wait_pending_submits()
        if byte_offset < 0:
            raise ValueError(f"Buffer readback byte_offset must be non-negative, got {byte_offset}")
        resolved_size = buffer.size - byte_offset if size is None else int(size)
        if resolved_size < 0:
            raise ValueError(f"Buffer readback size must be non-negative, got {resolved_size}")
        if byte_offset + resolved_size > buffer.size:
            raise ValueError(
                f"Buffer readback range [{byte_offset}, {byte_offset + resolved_size}) exceeds buffer size {buffer.size}"
            )
        staging_allocation = self.memory_manager.allocate_host_readback_buffer(
            resolved_size,
            usage_flags=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        )
        try:
            self.copy_buffer(
                buffer,
                staging_allocation.buffer,
                resolved_size,
                src_offset=byte_offset,
                dst_offset=staging_allocation.offset,
            )
            self.memory_manager.host_readback_ring.invalidate(allocation=staging_allocation)
            return staging_allocation.buffer.read_bytes_at(staging_allocation.offset, resolved_size)
        finally:
            staging_allocation.close()

    def copy_buffer(
        self,
        src: BufferOwner,
        dst: BufferOwner,
        size: int,
        *,
        src_offset: int = 0,
        dst_offset: int = 0,
    ) -> None:
        self.copy_buffer_transfers(src, ((src_offset, dst, dst_offset, size),))

    def copy_buffer_transfers(
        self,
        src: BufferOwner,
        transfers: tuple[tuple[int, BufferOwner, int, int], ...] | list[tuple[int, BufferOwner, int, int]],
    ) -> None:
        self.require_open()
        if not transfers:
            return
        self.wait_pending_submits()
        command_buffer = self.allocate_command_buffer()
        vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo())
        for src_offset, dst, dst_offset, size in transfers:
            if src_offset < 0 or dst_offset < 0 or size < 0:
                raise ValueError(
                    f"Buffer copy requires non-negative offsets and size, got "
                    f"src_offset={src_offset}, dst_offset={dst_offset}, size={size}"
                )
            if src_offset + size > src.size:
                raise ValueError(
                    f"Source copy range [{src_offset}, {src_offset + size}) exceeds buffer size {src.size}"
                )
            if dst_offset + size > dst.size:
                raise ValueError(
                    f"Destination copy range [{dst_offset}, {dst_offset + size}) exceeds buffer size {dst.size}"
                )
            vkCmdCopyBuffer(
                command_buffer,
                src.handle,
                dst.handle,
                1,
                [VkBufferCopy(srcOffset=src_offset, dstOffset=dst_offset, size=size)],
            )
        vkEndCommandBuffer(command_buffer)
        self.submit_one_shot_and_wait(command_buffer)

    def zero_tensors(self, tensors: Sequence[BufferSlice]) -> None:
        self.require_open()
        if not tensors:
            return
        self.wait_pending_submits()
        command_buffer = self.allocate_command_buffer()
        vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo())
        for slice in tensors:
            if slice.offset % 4 != 0 or slice.nbytes % 4 != 0:
                raise ValueError(
                    f"vkCmdFillBuffer zero requires 4-byte aligned tensor range, got "
                    f"offset={slice.offset}, nbytes={slice.nbytes}"
                )
            vkCmdFillBuffer(command_buffer, slice.allocation.buffer.handle, slice.offset, slice.nbytes, 0)
        vkEndCommandBuffer(command_buffer)
        self.submit_one_shot_and_wait(command_buffer)

    def allocate_command_buffer(self) -> object:
        self.require_open()
        return vkAllocateCommandBuffers(
            self.device,
            VkCommandBufferAllocateInfo(
                commandPool=self.command_pool,
                level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]

    def free_command_buffer(self, command_buffer: object) -> None:
        self.require_open()
        vkFreeCommandBuffers(self.device, self.command_pool, 1, [command_buffer])

    def submit_one_shot_and_wait(self, command_buffer: object) -> None:
        self.require_open()
        self.wait_pending_submits()
        submit_point = self.memory_manager.note_queue_submit_started()
        submit_one_shot_and_wait(
            device_handle=self.device,
            queue_handle=self.queue,
            command_buffer=command_buffer,
            release_command_buffer=self.free_command_buffer,
        )
        self.memory_manager.note_queue_submit_completed(submit_point)

    def submit_and_wait(self, command_buffer: object) -> None:
        self.require_open()
        self.wait_pending_submits()
        submit_point = self.memory_manager.note_queue_submit_started()
        submit_and_wait(
            device_handle=self.device,
            queue_handle=self.queue,
            command_buffer=command_buffer,
        )
        self.memory_manager.note_queue_submit_completed(submit_point)

    def submit_and_wait_with_fence(self, command_buffer: object, fence: object) -> None:
        self.require_open()
        self.wait_pending_submits()
        submit_point = self.memory_manager.note_queue_submit_started()
        submit_and_wait_with_fence(
            device_handle=self.device,
            queue_handle=self.queue,
            command_buffer=command_buffer,
            fence=fence,
        )
        self.memory_manager.note_queue_submit_completed(submit_point)

    def submit_one_shot_async(self, command_buffer: object, *, cleanup: Callable[[], None]) -> int:
        return self.submit_command_buffer_async(
            command_buffer,
            cleanup=cleanup,
            release_command_buffer=True,
            destroy_fence=True,
        )

    def submit_command_buffer_async(
        self,
        command_buffer: object,
        *,
        fence: object | None = None,
        cleanup: Callable[[], None] | None = None,
        release_command_buffer: bool = False,
        destroy_fence: bool = False,
    ) -> int:
        self.require_open()
        throttle_wait_ns = 0
        if len(self._pending_submits) >= self._max_pending_submits:
            wait_started_ns = time.perf_counter_ns()
            self._wait_oldest_pending_submit()
            throttle_wait_ns = time.perf_counter_ns() - wait_started_ns
        cleanup_fn = _noop if cleanup is None else cleanup
        submit_id = self.memory_manager.note_queue_submit_started()
        owns_fence = destroy_fence
        try:
            if fence is None:
                fence = submit_with_new_fence(
                    device_handle=self.device,
                    queue_handle=self.queue,
                    command_buffer=command_buffer,
                )
                owns_fence = True
            else:
                submit_with_existing_fence(
                    device_handle=self.device,
                    queue_handle=self.queue,
                    command_buffer=command_buffer,
                    fence=fence,
                )
        except Exception:
            self.memory_manager.note_queue_submit_completed(submit_id)
            if release_command_buffer:
                self.free_command_buffer(command_buffer)
            cleanup_fn()
            raise
        self._pending_submits.append(
            _PendingCommandBufferSubmit(
                submit_id=submit_id,
                command_buffer=command_buffer,
                fence=fence,
                cleanup=cleanup_fn,
                release_command_buffer=release_command_buffer,
                destroy_fence=owns_fence,
            )
        )
        return throttle_wait_ns

    def wait_pending_submits(self) -> None:
        self.require_open()
        while self._pending_submits:
            self._wait_oldest_pending_submit()

    def _wait_oldest_pending_submit(self) -> None:
        submit = self._pending_submits[0]
        wait_for_fence(device_handle=self.device, fence=submit.fence)
        self._pending_submits.pop(0)
        try:
            if submit.release_command_buffer:
                self.free_command_buffer(submit.command_buffer)
            submit.cleanup()
        finally:
            if submit.destroy_fence:
                vkDestroyFence(self.device, submit.fence, None)
            self.memory_manager.note_queue_submit_completed(submit.submit_id)


def _noop() -> None:
    return None
