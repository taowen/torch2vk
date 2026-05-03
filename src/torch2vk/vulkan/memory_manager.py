"""Centralized Vulkan memory ownership and allocation orchestration."""

from __future__ import annotations

import importlib
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from vulkan import (
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
)
from vulkan._vulkan import ffi

from torch2vk.vulkan.types import TensorSpec

from .abi import VkPhysicalDeviceMemoryProperties

from .allocation import BufferAllocation, BufferSlice, GpuTimelinePoint
from .memory_allocator import VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_VALUE, create_buffer, tensor_nbytes


class _VkFlushMappedMemoryRanges(Protocol):
    def __call__(self, device: object, memory_range_count: int, memory_ranges: object) -> int: ...


class _VkInvalidateMappedMemoryRanges(Protocol):
    def __call__(self, device: object, memory_range_count: int, memory_ranges: object) -> int: ...


_vulkan_cffi = importlib.import_module("vulkan._vulkan")
_vulkan_lib = getattr(_vulkan_cffi, "lib")
_ffi_null = ffi.NULL
def _flush_mapped_memory_ranges_proc() -> _VkFlushMappedMemoryRanges:
    fn = getattr(_vulkan_lib, "vkFlushMappedMemoryRanges")
    if not callable(fn):
        raise TypeError("vkFlushMappedMemoryRanges is not callable")

    def flush(device: object, memory_range_count: int, memory_ranges: object) -> int:
        result = fn(device, memory_range_count, memory_ranges)
        if not isinstance(result, int):
            raise TypeError(f"vkFlushMappedMemoryRanges returned {type(result).__name__}")
        return result

    return flush


def _invalidate_mapped_memory_ranges_proc() -> _VkInvalidateMappedMemoryRanges:
    fn = getattr(_vulkan_lib, "vkInvalidateMappedMemoryRanges")
    if not callable(fn):
        raise TypeError("vkInvalidateMappedMemoryRanges is not callable")

    def invalidate(device: object, memory_range_count: int, memory_ranges: object) -> int:
        result = fn(device, memory_range_count, memory_ranges)
        if not isinstance(result, int):
            raise TypeError(f"vkInvalidateMappedMemoryRanges returned {type(result).__name__}")
        return result

    return invalidate


_vk_flush_mapped_memory_ranges = _flush_mapped_memory_ranges_proc()
_vk_invalidate_mapped_memory_ranges = _invalidate_mapped_memory_ranges_proc()


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((int(value) + alignment - 1) // alignment) * alignment


def _slice_for_spec(*, spec: TensorSpec, allocation: BufferAllocation) -> BufferSlice:
    return BufferSlice(allocation=allocation, offset=allocation.offset, nbytes=tensor_nbytes(spec))


@dataclass(frozen=True, slots=True)
class MemoryAllocationStats:
    device_local_allocations: int = 0
    host_upload_allocations: int = 0
    host_readback_allocations: int = 0
    host_visible_allocations: int = 0
    device_local_live_bytes: int = 0
    host_upload_live_bytes: int = 0
    host_readback_live_bytes: int = 0
    host_visible_live_bytes: int = 0
    device_local_peak_live_bytes: int = 0
    host_upload_peak_live_bytes: int = 0
    host_readback_peak_live_bytes: int = 0
    host_visible_peak_live_bytes: int = 0
    device_local_reserved_bytes: int = 0
    host_upload_reserved_bytes: int = 0
    host_readback_reserved_bytes: int = 0
    host_visible_reserved_bytes: int = 0
    device_local_peak_reserved_bytes: int = 0
    host_upload_peak_reserved_bytes: int = 0
    host_readback_peak_reserved_bytes: int = 0
    host_visible_peak_reserved_bytes: int = 0
    peak_total_live_bytes: int = 0
    peak_total_reserved_bytes: int = 0

    def delta_from(self, baseline: "MemoryAllocationStats") -> "MemoryAllocationStats":
        return MemoryAllocationStats(
            device_local_allocations=self.device_local_allocations - baseline.device_local_allocations,
            host_upload_allocations=self.host_upload_allocations - baseline.host_upload_allocations,
            host_readback_allocations=self.host_readback_allocations - baseline.host_readback_allocations,
            host_visible_allocations=self.host_visible_allocations - baseline.host_visible_allocations,
            device_local_live_bytes=self.device_local_live_bytes - baseline.device_local_live_bytes,
            host_upload_live_bytes=self.host_upload_live_bytes - baseline.host_upload_live_bytes,
            host_readback_live_bytes=self.host_readback_live_bytes - baseline.host_readback_live_bytes,
            host_visible_live_bytes=self.host_visible_live_bytes - baseline.host_visible_live_bytes,
            device_local_peak_live_bytes=max(
                0,
                self.device_local_peak_live_bytes - baseline.device_local_live_bytes,
            ),
            host_upload_peak_live_bytes=max(
                0,
                self.host_upload_peak_live_bytes - baseline.host_upload_live_bytes,
            ),
            host_readback_peak_live_bytes=max(
                0,
                self.host_readback_peak_live_bytes - baseline.host_readback_live_bytes,
            ),
            host_visible_peak_live_bytes=max(
                0,
                self.host_visible_peak_live_bytes - baseline.host_visible_live_bytes,
            ),
            device_local_reserved_bytes=self.device_local_reserved_bytes - baseline.device_local_reserved_bytes,
            host_upload_reserved_bytes=self.host_upload_reserved_bytes - baseline.host_upload_reserved_bytes,
            host_readback_reserved_bytes=self.host_readback_reserved_bytes - baseline.host_readback_reserved_bytes,
            host_visible_reserved_bytes=self.host_visible_reserved_bytes - baseline.host_visible_reserved_bytes,
            device_local_peak_reserved_bytes=max(
                0,
                self.device_local_peak_reserved_bytes - baseline.device_local_reserved_bytes,
            ),
            host_upload_peak_reserved_bytes=max(
                0,
                self.host_upload_peak_reserved_bytes - baseline.host_upload_reserved_bytes,
            ),
            host_readback_peak_reserved_bytes=max(
                0,
                self.host_readback_peak_reserved_bytes - baseline.host_readback_reserved_bytes,
            ),
            host_visible_peak_reserved_bytes=max(
                0,
                self.host_visible_peak_reserved_bytes - baseline.host_visible_reserved_bytes,
            ),
            peak_total_live_bytes=max(0, self.peak_total_live_bytes - baseline.total_live_bytes()),
            peak_total_reserved_bytes=max(0, self.peak_total_reserved_bytes - baseline.total_reserved_bytes()),
        )

    def total(self) -> int:
        return (
            self.device_local_allocations
            + self.host_upload_allocations
            + self.host_readback_allocations
            + self.host_visible_allocations
        )

    def total_live_bytes(self) -> int:
        return (
            self.device_local_live_bytes
            + self.host_upload_live_bytes
            + self.host_readback_live_bytes
            + self.host_visible_live_bytes
        )

    def total_peak_live_bytes(self) -> int:
        return (
            self.device_local_peak_live_bytes
            + self.host_upload_peak_live_bytes
            + self.host_readback_peak_live_bytes
            + self.host_visible_peak_live_bytes
        )

    def total_reserved_bytes(self) -> int:
        return (
            self.device_local_reserved_bytes
            + self.host_upload_reserved_bytes
            + self.host_readback_reserved_bytes
            + self.host_visible_reserved_bytes
        )

    def total_peak_reserved_bytes(self) -> int:
        return (
            self.device_local_peak_reserved_bytes
            + self.host_upload_peak_reserved_bytes
            + self.host_readback_peak_reserved_bytes
            + self.host_visible_peak_reserved_bytes
        )


@dataclass(slots=True)
class _ArenaChunk:
    root: BufferAllocation
    free_ranges: list[tuple[int, int]]


class DeviceLocalArena:
    """Owns long-lived device-local allocations with chunk suballocation."""

    def __init__(
        self,
        *,
        device_handle: object,
        memory_properties: VkPhysicalDeviceMemoryProperties,
        require_device_open: Callable[[], None],
        is_device_closed: Callable[[], bool],
        chunk_size: int = 64 * 1024 * 1024,
        alignment: int = 256,
    ) -> None:
        self._device_handle = device_handle
        self._memory_properties = memory_properties
        self._require_device_open = require_device_open
        self._is_device_closed = is_device_closed
        self._chunk_size = int(chunk_size)
        self._alignment = int(alignment)
        self._chunks_by_usage: dict[int, list[_ArenaChunk]] = defaultdict(list)
        self._roots: list[BufferAllocation] = []
        self._closed = False

    @property
    def reserved_bytes(self) -> int:
        return sum(root.size for root in self._roots if not root.released)

    def _allocate_root(self, *, size: int, usage_flags: int, pool: str) -> BufferAllocation:
        buffer = create_buffer(
            device_handle=self._device_handle,
            memory_properties=self._memory_properties,
            require_device_open=self._require_device_open,
            is_device_closed=self._is_device_closed,
            size=size,
            usage_flags=usage_flags,
            memory_property_flags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        root = BufferAllocation(buffer=buffer, pool=pool, offset=0, size_bytes=size, vk_allocation=True)
        self._roots.append(root)
        return root

    @staticmethod
    def _coalesce(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not ranges:
            return []
        sorted_ranges = sorted(ranges)
        merged: list[tuple[int, int]] = []
        cur_start, cur_size = sorted_ranges[0]
        cur_end = cur_start + cur_size
        for start, size in sorted_ranges[1:]:
            end = start + size
            if start <= cur_end:
                cur_end = max(cur_end, end)
                continue
            merged.append((cur_start, cur_end - cur_start))
            cur_start, cur_end = start, end
        merged.append((cur_start, cur_end - cur_start))
        return merged

    def _release_subrange(self, *, chunk: _ArenaChunk, local_offset: int, size: int) -> None:
        chunk.free_ranges.append((local_offset, size))
        chunk.free_ranges = self._coalesce(chunk.free_ranges)

    def _suballocate_from_chunk(self, *, chunk: _ArenaChunk, size: int, pool: str) -> BufferAllocation | None:
        needed = _align_up(size, self._alignment)
        for index, (range_offset, range_size) in enumerate(chunk.free_ranges):
            aligned_offset = _align_up(range_offset, self._alignment)
            prefix = aligned_offset - range_offset
            usable = range_size - prefix
            if usable < needed:
                continue
            remainder_start = aligned_offset + needed
            remainder_size = (range_offset + range_size) - remainder_start
            replacement: list[tuple[int, int]] = []
            if prefix > 0:
                replacement.append((range_offset, prefix))
            if remainder_size > 0:
                replacement.append((remainder_start, remainder_size))
            chunk.free_ranges[index : index + 1] = replacement
            absolute_offset = chunk.root.offset + aligned_offset
            return BufferAllocation(
                buffer=chunk.root.buffer,
                pool=pool,
                offset=absolute_offset,
                size_bytes=size,
                releaser=lambda _allocation: self._release_subrange(
                    chunk=chunk,
                    local_offset=aligned_offset,
                    size=needed,
                ),
            )
        return None

    def allocate(self, *, size: int, usage_flags: int, pool: str = "device_local") -> BufferAllocation:
        if self._closed:
            raise RuntimeError("DeviceLocalArena is closed")
        if size <= 0:
            raise ValueError(f"DeviceLocalArena size must be positive, got {size}")
        aligned_size = _align_up(size, self._alignment)
        for chunk in self._chunks_by_usage[usage_flags]:
            allocation = self._suballocate_from_chunk(chunk=chunk, size=size, pool=pool)
            if allocation is not None:
                return allocation
        chunk_bytes = max(_align_up(self._chunk_size, self._alignment), aligned_size)
        root = self._allocate_root(size=chunk_bytes, usage_flags=usage_flags, pool=f"{pool}:chunk")
        new_chunk = _ArenaChunk(root=root, free_ranges=[(0, chunk_bytes)])
        self._chunks_by_usage[usage_flags].append(new_chunk)
        allocation = self._suballocate_from_chunk(chunk=new_chunk, size=size, pool=pool)
        if allocation is None:
            raise RuntimeError("DeviceLocalArena failed to suballocate from newly created chunk")
        return allocation

    def allocate_dedicated(self, *, size: int, usage_flags: int, pool: str = "device_local:dedicated") -> BufferAllocation:
        if self._closed:
            raise RuntimeError("DeviceLocalArena is closed")
        if size <= 0:
            raise ValueError(f"DeviceLocalArena dedicated size must be positive, got {size}")
        return self._allocate_root(size=size, usage_flags=usage_flags, pool=pool)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while self._roots:
            self._roots.pop().close()
        self._chunks_by_usage.clear()


@dataclass(slots=True)
class _HostRingSegment:
    root: BufferAllocation
    cursor: int = 0


class HostRing:
    """Owns host-visible ring allocations for upload/readback staging."""

    def __init__(
        self,
        *,
        ring_name: str,
        device_handle: object,
        memory_properties: VkPhysicalDeviceMemoryProperties,
        require_device_open: Callable[[], None],
        is_device_closed: Callable[[], bool],
        segment_size: int = 64 * 1024 * 1024,
        alignment: int = 256,
    ) -> None:
        self._ring_name = ring_name
        self._device_handle = device_handle
        self._memory_properties = memory_properties
        self._require_device_open = require_device_open
        self._is_device_closed = is_device_closed
        self._segment_size = int(segment_size)
        self._alignment = int(alignment)
        self._segments: dict[int, list[_HostRingSegment]] = defaultdict(list)
        self._roots: list[BufferAllocation] = []
        self._closed = False

    @property
    def reserved_bytes(self) -> int:
        return sum(root.size for root in self._roots if not root.released)

    def _new_segment(self, *, usage_flags: int, minimum_bytes: int) -> _HostRingSegment:
        segment_bytes = max(_align_up(self._segment_size, self._alignment), _align_up(minimum_bytes, self._alignment))
        root = BufferAllocation(
            buffer=create_buffer(
                device_handle=self._device_handle,
                memory_properties=self._memory_properties,
                require_device_open=self._require_device_open,
                is_device_closed=self._is_device_closed,
                size=segment_bytes,
                usage_flags=usage_flags,
                memory_property_flags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ),
            pool=f"{self._ring_name}:segment",
            offset=0,
            size_bytes=segment_bytes,
            vk_allocation=True,
        )
        root.buffer.map_persistent()
        self._roots.append(root)
        segment = _HostRingSegment(root=root, cursor=0)
        self._segments[usage_flags].append(segment)
        return segment

    def _allocate_dedicated(self, *, size: int, usage_flags: int) -> BufferAllocation:
        root = BufferAllocation(
            buffer=create_buffer(
                device_handle=self._device_handle,
                memory_properties=self._memory_properties,
                require_device_open=self._require_device_open,
                is_device_closed=self._is_device_closed,
                size=size,
                usage_flags=usage_flags,
                memory_property_flags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ),
            pool=f"{self._ring_name}:dedicated",
            offset=0,
            size_bytes=size,
            vk_allocation=True,
        )
        root.buffer.map_persistent()
        self._roots.append(root)
        return BufferAllocation(
            buffer=root.buffer,
            pool=self._ring_name,
            offset=0,
            size_bytes=size,
            vk_allocation=True,
            releaser=lambda _allocation: root.close(),
        )

    def allocate(self, *, size: int, usage_flags: int) -> BufferAllocation:
        if self._closed:
            raise RuntimeError(f"{self._ring_name} is closed")
        if size <= 0:
            raise ValueError(f"{self._ring_name} allocation size must be positive, got {size}")
        needed = _align_up(size, self._alignment)
        if needed > self._segment_size:
            return self._allocate_dedicated(size=needed, usage_flags=usage_flags)
        segments = self._segments[usage_flags]
        if not segments:
            segments.append(self._new_segment(usage_flags=usage_flags, minimum_bytes=needed))
        segment = segments[-1]
        if segment.cursor + needed > segment.root.size:
            segment.cursor = 0
        if segment.cursor + needed > segment.root.size:
            segment = self._new_segment(usage_flags=usage_flags, minimum_bytes=needed)
        absolute_offset = segment.root.offset + segment.cursor
        segment.cursor += needed
        return BufferAllocation(
            buffer=segment.root.buffer,
            pool=self._ring_name,
            offset=absolute_offset,
            size_bytes=size,
            releaser=lambda _allocation: None,
        )

    def flush(self, *, allocation: BufferAllocation, byte_offset: int = 0, size: int | None = None) -> None:
        start = allocation.offset + int(byte_offset)
        if start < allocation.offset:
            raise ValueError(f"{self._ring_name} flush start out of range: {start}")
        nbytes = allocation.size - int(byte_offset) if size is None else int(size)
        if nbytes < 0 or start + nbytes > allocation.end_offset:
            raise ValueError(f"{self._ring_name} flush range exceeds allocation bounds")
        self._require_device_open()
        mapped_range = ffi.new(
            "VkMappedMemoryRange *",
            {
                "sType": 6,
                "pNext": _ffi_null,
                "memory": allocation.buffer.memory,
                "offset": start,
                "size": nbytes,
            },
        )
        _vk_flush_mapped_memory_ranges(self._device_handle, 1, mapped_range)

    def invalidate(self, *, allocation: BufferAllocation, byte_offset: int = 0, size: int | None = None) -> None:
        start = allocation.offset + int(byte_offset)
        if start < allocation.offset:
            raise ValueError(f"{self._ring_name} invalidate start out of range: {start}")
        nbytes = allocation.size - int(byte_offset) if size is None else int(size)
        if nbytes < 0 or start + nbytes > allocation.end_offset:
            raise ValueError(f"{self._ring_name} invalidate range exceeds allocation bounds")
        self._require_device_open()
        mapped_range = ffi.new(
            "VkMappedMemoryRange *",
            {
                "sType": 6,
                "pNext": _ffi_null,
                "memory": allocation.buffer.memory,
                "offset": start,
                "size": nbytes,
            },
        )
        _vk_invalidate_mapped_memory_ranges(self._device_handle, 1, mapped_range)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while self._roots:
            self._roots.pop().close()
        self._segments.clear()


class DedicatedAllocationPolicy:
    """Central policy for dedicated allocation decisions."""

    def should_dedicate(self, *, size: int) -> bool:
        return size >= 128 * 1024 * 1024


class RetiredAllocationQueue:
    """Tracks allocations pending safe recycling after GPU completion."""

    def __init__(self) -> None:
        self._retired: deque[tuple[TensorSpec, BufferAllocation]] = deque()

    def retire(self, *, spec: TensorSpec, allocation: BufferAllocation, last_use: GpuTimelinePoint | None) -> None:
        allocation.last_use = last_use
        self._retired.append((spec, allocation))

    def pop_reclaimable(self, *, completed: GpuTimelinePoint | None) -> tuple[tuple[TensorSpec, BufferAllocation], ...]:
        if not self._retired:
            return ()
        if completed is None:
            reclaimable = tuple(self._retired)
            self._retired.clear()
            return reclaimable
        ready: list[tuple[TensorSpec, BufferAllocation]] = []
        pending: deque[tuple[TensorSpec, BufferAllocation]] = deque()
        while self._retired:
            spec, allocation = self._retired.popleft()
            last_use = allocation.last_use
            if (
                last_use is None
                or (
                    last_use.queue_id == completed.queue_id
                    and last_use.submit_id <= completed.submit_id
                )
            ):
                ready.append((spec, allocation))
            else:
                pending.append((spec, allocation))
        self._retired = pending
        return tuple(ready)

    def close(self) -> None:
        while self._retired:
            _spec, allocation = self._retired.pop()
            allocation.close()


class TemporaryTensorPool:
    """Reusable temporary tensor allocation pool with exact-shape then size-class reuse."""

    def __init__(
        self,
        *,
        allocate_device_local: Callable[[int], BufferAllocation],
        retire_allocation: Callable[[TensorSpec, BufferAllocation], None],
        reclaim_retired: Callable[[], None],
        note_reuse: Callable[[BufferAllocation], None],
        note_recycle: Callable[[BufferAllocation], None],
    ) -> None:
        self._allocate_device_local = allocate_device_local
        self._retire_allocation = retire_allocation
        self._reclaim_retired = reclaim_retired
        self._note_reuse = note_reuse
        self._note_recycle = note_recycle
        self._free_exact: dict[tuple[str, tuple[int, ...]], list[BufferAllocation]] = defaultdict(list)
        self._free_size_class: dict[tuple[str, int], list[BufferAllocation]] = defaultdict(list)
        self._recycled_spec_by_allocation_id: dict[int, tuple[str, tuple[int, ...]]] = {}
        self._closed = False

    @staticmethod
    def _size_class(bytes_size: int) -> int:
        value = max(int(bytes_size), 1024)
        return 1 << (value - 1).bit_length()

    def acquire(
        self,
        spec: TensorSpec,
        *,
        label: str | None = None,
        allocate: Callable[[TensorSpec, str | None], tuple[BufferSlice, BufferAllocation]] | None = None,
    ) -> tuple[BufferSlice, BufferAllocation]:
        if self._closed:
            raise RuntimeError("TemporaryTensorPool is closed")
        if allocate is not None:
            return allocate(spec, label)
        self._reclaim_retired()
        exact_key = (spec.dtype, tuple(int(dim) for dim in spec.shape))
        bucket = self._free_exact.get(exact_key)
        if bucket:
            allocation = bucket.pop()
            self._recycled_spec_by_allocation_id.pop(id(allocation), None)
            self._note_reuse(allocation)
            return _slice_for_spec(spec=spec, allocation=allocation), allocation

        requested_nbytes = tensor_nbytes(spec)
        class_key = (spec.dtype, self._size_class(requested_nbytes))
        class_bucket = self._free_size_class.get(class_key)
        if class_bucket:
            allocation = class_bucket.pop()
            recycled_key = self._recycled_spec_by_allocation_id.pop(id(allocation), None)
            if recycled_key is not None:
                recycled_bucket = self._free_exact.get(recycled_key)
                if recycled_bucket:
                    for index in range(len(recycled_bucket) - 1, -1, -1):
                        if recycled_bucket[index] is allocation:
                            del recycled_bucket[index]
                            break
            self._note_reuse(allocation)
            return _slice_for_spec(spec=spec, allocation=allocation), allocation

        allocation = self._allocate_device_local(requested_nbytes)
        return _slice_for_spec(spec=spec, allocation=allocation), allocation

    def recycle(self, *, spec: TensorSpec, allocation: BufferAllocation) -> None:
        if self._closed:
            allocation.close()
            return
        self._note_recycle(allocation)
        self._retire_allocation(spec, allocation)

    def reclaim(self, *, spec: TensorSpec, allocation: BufferAllocation) -> None:
        if self._closed:
            allocation.close()
            return
        exact_key = (spec.dtype, tuple(int(dim) for dim in spec.shape))
        self._free_exact[exact_key].append(allocation)
        class_key = (spec.dtype, self._size_class(allocation.size))
        self._free_size_class[class_key].append(allocation)
        self._recycled_spec_by_allocation_id[id(allocation)] = exact_key

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        closed_ids: set[int] = set()
        for bucket in self._free_exact.values():
            while bucket:
                allocation = bucket.pop()
                allocation_id = id(allocation)
                if allocation_id in closed_ids:
                    continue
                closed_ids.add(allocation_id)
                allocation.close()
        self._free_exact.clear()
        self._free_size_class.clear()
        self._recycled_spec_by_allocation_id.clear()


class MemoryManager:
    """Aggregates model-agnostic memory ownership components."""

    def __init__(
        self,
        *,
        device_handle: object,
        memory_properties: VkPhysicalDeviceMemoryProperties,
        require_device_open: Callable[[], None],
        is_device_closed: Callable[[], bool],
    ) -> None:
        self._queue_id = "compute:0"
        self._last_submitted_submit_id = 0
        self._last_completed_submit_id = 0
        self.device_local_arena = DeviceLocalArena(
            device_handle=device_handle,
            memory_properties=memory_properties,
            require_device_open=require_device_open,
            is_device_closed=is_device_closed,
        )
        self.host_upload_ring = HostRing(
            ring_name="host_upload",
            device_handle=device_handle,
            memory_properties=memory_properties,
            require_device_open=require_device_open,
            is_device_closed=is_device_closed,
        )
        self.host_readback_ring = HostRing(
            ring_name="host_readback",
            device_handle=device_handle,
            memory_properties=memory_properties,
            require_device_open=require_device_open,
            is_device_closed=is_device_closed,
        )
        self.dedicated_policy = DedicatedAllocationPolicy()
        self.retired_queue = RetiredAllocationQueue()
        self.temporary_tensor_pool = TemporaryTensorPool(
            allocate_device_local=self.allocate_device_local_buffer,
            retire_allocation=self._retire_temporary_allocation,
            reclaim_retired=self.reclaim_retired_allocations,
            note_reuse=lambda allocation: self._note_allocation_reused("device_local", allocation.size),
            note_recycle=lambda allocation: self._note_allocation_recycled("device_local", allocation.size),
        )
        self._allocation_stats = MemoryAllocationStats()
        self._allocation_epoch = 0
        self._closed = False

    def _submitted_timeline_point(self) -> GpuTimelinePoint:
        return GpuTimelinePoint(
            queue_id=self._queue_id,
            submit_id=self._last_submitted_submit_id,
            fence=None,
        )

    def _completed_timeline_point(self) -> GpuTimelinePoint:
        return GpuTimelinePoint(
            queue_id=self._queue_id,
            submit_id=self._last_completed_submit_id,
            fence=None,
        )

    def note_queue_submit_started(self) -> int:
        self._last_submitted_submit_id += 1
        return self._last_submitted_submit_id

    def note_queue_submit_completed(self, submit_id: int | None = None) -> GpuTimelinePoint:
        resolved_submit_id = self._last_submitted_submit_id if submit_id is None else int(submit_id)
        if resolved_submit_id > self._last_submitted_submit_id:
            raise ValueError(
                f"Cannot complete submit_id={resolved_submit_id} before submission "
                f"(last_submitted={self._last_submitted_submit_id})"
            )
        self._last_completed_submit_id = max(self._last_completed_submit_id, resolved_submit_id)
        self.reclaim_retired_allocations()
        return self._completed_timeline_point()

    def note_queue_progress(self, *, completed_submit_id: int) -> GpuTimelinePoint:
        if completed_submit_id < 0:
            raise ValueError(f"completed_submit_id must be non-negative, got {completed_submit_id}")
        if completed_submit_id > self._last_submitted_submit_id:
            raise ValueError(
                f"completed_submit_id={completed_submit_id} exceeds last_submitted={self._last_submitted_submit_id}"
            )
        self._last_completed_submit_id = max(self._last_completed_submit_id, int(completed_submit_id))
        self.reclaim_retired_allocations()
        return self._completed_timeline_point()

    def _retire_temporary_allocation(self, spec: TensorSpec, allocation: BufferAllocation) -> None:
        self.retired_queue.retire(
            spec=spec,
            allocation=allocation,
            last_use=self._submitted_timeline_point(),
        )
        self.reclaim_retired_allocations()

    def reclaim_retired_allocations(self) -> None:
        completed = self._completed_timeline_point()
        for spec, allocation in self.retired_queue.pop_reclaimable(completed=completed):
            self.temporary_tensor_pool.reclaim(spec=spec, allocation=allocation)

    def _note_allocation(self, pool: str, allocation: BufferAllocation) -> BufferAllocation:
        previous_reserved = _stats_reserved_bytes(self._allocation_stats, pool)
        stats_with_reserved = self._stats_with_reserved_bytes(self._allocation_stats)
        reserved_grew = _stats_reserved_bytes(stats_with_reserved, pool) > previous_reserved
        if allocation.vk_allocation or reserved_grew:
            self._allocation_epoch += 1
        self._allocation_stats = _stats_add_live_bytes(
            stats_with_reserved,
            pool=pool,
            allocation_count=1 if allocation.vk_allocation else 0,
            live_bytes=allocation.size,
        )
        return _wrap_allocation_releaser(
            allocation,
            release_hook=lambda released: self._note_allocation_released(pool, released.size),
        )

    def _note_allocation_released(self, pool: str, size: int) -> None:
        if self._closed:
            return
        self._allocation_stats = _stats_subtract_live_bytes(
            self._allocation_stats,
            pool=pool,
            live_bytes=size,
        )
        self._allocation_stats = self._stats_with_reserved_bytes(self._allocation_stats)

    def _note_allocation_reused(self, pool: str, size: int) -> None:
        self._allocation_stats = _stats_add_live_bytes(
            self._allocation_stats,
            pool=pool,
            allocation_count=0,
            live_bytes=size,
        )

    def _note_allocation_recycled(self, pool: str, size: int) -> None:
        self._allocation_stats = _stats_subtract_live_bytes(
            self._allocation_stats,
            pool=pool,
            live_bytes=size,
        )

    def _stats_with_reserved_bytes(self, stats: MemoryAllocationStats) -> MemoryAllocationStats:
        device_local_reserved = self.device_local_arena.reserved_bytes
        host_upload_reserved = self.host_upload_ring.reserved_bytes
        host_readback_reserved = self.host_readback_ring.reserved_bytes
        host_visible_reserved = 0
        return _stats_with_peak_total_reserved_bytes(replace(
            stats,
            device_local_reserved_bytes=device_local_reserved,
            host_upload_reserved_bytes=host_upload_reserved,
            host_readback_reserved_bytes=host_readback_reserved,
            host_visible_reserved_bytes=host_visible_reserved,
            device_local_peak_reserved_bytes=max(
                stats.device_local_peak_reserved_bytes,
                device_local_reserved,
            ),
            host_upload_peak_reserved_bytes=max(
                stats.host_upload_peak_reserved_bytes,
                host_upload_reserved,
            ),
            host_readback_peak_reserved_bytes=max(
                stats.host_readback_peak_reserved_bytes,
                host_readback_reserved,
            ),
            host_visible_peak_reserved_bytes=max(
                stats.host_visible_peak_reserved_bytes,
                host_visible_reserved,
            ),
        ))

    def allocate_device_local_buffer(self, size: int, *, usage_flags: int | None = None) -> BufferAllocation:
        resolved_usage_flags = (
            (
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_VALUE
            )
            if usage_flags is None
            else int(usage_flags)
        )
        if resolved_usage_flags <= 0:
            raise ValueError(f"Device-local usage_flags must be positive, got {resolved_usage_flags}")
        if self.dedicated_policy.should_dedicate(size=size):
            allocation = self.device_local_arena.allocate_dedicated(size=size, usage_flags=resolved_usage_flags)
        else:
            allocation = self.device_local_arena.allocate(size=size, usage_flags=resolved_usage_flags)
        return self._note_allocation("device_local", allocation)

    def allocate_host_upload_buffer(self, size: int, *, usage_flags: int | None = None) -> BufferAllocation:
        resolved_usage_flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT if usage_flags is None else int(usage_flags)
        if resolved_usage_flags <= 0:
            raise ValueError(f"Host-upload usage_flags must be positive, got {resolved_usage_flags}")
        allocation = self.host_upload_ring.allocate(size=size, usage_flags=resolved_usage_flags)
        return self._note_allocation("host_upload", allocation)

    def allocate_host_readback_buffer(self, size: int, *, usage_flags: int | None = None) -> BufferAllocation:
        resolved_usage_flags = VK_BUFFER_USAGE_TRANSFER_DST_BIT if usage_flags is None else int(usage_flags)
        if resolved_usage_flags <= 0:
            raise ValueError(f"Host-readback usage_flags must be positive, got {resolved_usage_flags}")
        allocation = self.host_readback_ring.allocate(size=size, usage_flags=resolved_usage_flags)
        return self._note_allocation("host_readback", allocation)

    def allocate_host_visible_buffer(self, size: int, *, usage_flags: int | None = None) -> BufferAllocation:
        resolved_usage_flags = (
            (
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_VALUE
            )
            if usage_flags is None
            else int(usage_flags)
        )
        if resolved_usage_flags <= 0:
            raise ValueError(f"Host-visible usage_flags must be positive, got {resolved_usage_flags}")
        allocation = self.host_upload_ring.allocate(size=size, usage_flags=resolved_usage_flags)
        return self._note_allocation("host_visible", allocation)

    def allocation_stats(self) -> MemoryAllocationStats:
        return self._allocation_stats

    def allocation_epoch(self) -> int:
        return self._allocation_epoch

    def reset_allocation_stats(self) -> None:
        self._allocation_stats = self._stats_with_reserved_bytes(MemoryAllocationStats())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.temporary_tensor_pool.close()
        self.retired_queue.close()
        self.host_readback_ring.close()
        self.host_upload_ring.close()
        self.device_local_arena.close()


def _wrap_allocation_releaser(
    allocation: BufferAllocation,
    *,
    release_hook: Callable[[BufferAllocation], None],
) -> BufferAllocation:
    original_releaser = allocation.releaser

    def _release(released: BufferAllocation) -> None:
        try:
            release_hook(released)
        finally:
            if original_releaser is not None:
                original_releaser(released)
            else:
                released.buffer.close()

    allocation.releaser = _release
    return allocation


def _stats_add_live_bytes(
    stats: MemoryAllocationStats,
    *,
    pool: str,
    allocation_count: int,
    live_bytes: int,
) -> MemoryAllocationStats:
    if pool == "device_local":
        current = stats.device_local_live_bytes + live_bytes
        return _stats_with_peak_total_live_bytes(replace(
            stats,
            device_local_allocations=stats.device_local_allocations + allocation_count,
            device_local_live_bytes=current,
            device_local_peak_live_bytes=max(stats.device_local_peak_live_bytes, current),
        ))
    if pool == "host_upload":
        current = stats.host_upload_live_bytes + live_bytes
        return _stats_with_peak_total_live_bytes(replace(
            stats,
            host_upload_allocations=stats.host_upload_allocations + allocation_count,
            host_upload_live_bytes=current,
            host_upload_peak_live_bytes=max(stats.host_upload_peak_live_bytes, current),
        ))
    if pool == "host_readback":
        current = stats.host_readback_live_bytes + live_bytes
        return _stats_with_peak_total_live_bytes(replace(
            stats,
            host_readback_allocations=stats.host_readback_allocations + allocation_count,
            host_readback_live_bytes=current,
            host_readback_peak_live_bytes=max(stats.host_readback_peak_live_bytes, current),
        ))
    if pool == "host_visible":
        current = stats.host_visible_live_bytes + live_bytes
        return _stats_with_peak_total_live_bytes(replace(
            stats,
            host_visible_allocations=stats.host_visible_allocations + allocation_count,
            host_visible_live_bytes=current,
            host_visible_peak_live_bytes=max(stats.host_visible_peak_live_bytes, current),
        ))
    raise ValueError(f"Unknown memory pool {pool!r}")


def _stats_subtract_live_bytes(
    stats: MemoryAllocationStats,
    *,
    pool: str,
    live_bytes: int,
) -> MemoryAllocationStats:
    if pool == "device_local":
        return replace(
            stats,
            device_local_live_bytes=max(0, stats.device_local_live_bytes - live_bytes),
        )
    if pool == "host_upload":
        return replace(
            stats,
            host_upload_live_bytes=max(0, stats.host_upload_live_bytes - live_bytes),
        )
    if pool == "host_readback":
        return replace(
            stats,
            host_readback_live_bytes=max(0, stats.host_readback_live_bytes - live_bytes),
        )
    if pool == "host_visible":
        return replace(
            stats,
            host_visible_live_bytes=max(0, stats.host_visible_live_bytes - live_bytes),
        )
    raise ValueError(f"Unknown memory pool {pool!r}")


def _stats_reserved_bytes(stats: MemoryAllocationStats, pool: str) -> int:
    if pool == "device_local":
        return stats.device_local_reserved_bytes
    if pool == "host_upload":
        return stats.host_upload_reserved_bytes
    if pool == "host_readback":
        return stats.host_readback_reserved_bytes
    if pool == "host_visible":
        return stats.host_visible_reserved_bytes
    raise ValueError(f"Unknown memory pool {pool!r}")


def _stats_with_peak_total_live_bytes(stats: MemoryAllocationStats) -> MemoryAllocationStats:
    return replace(stats, peak_total_live_bytes=max(stats.peak_total_live_bytes, stats.total_live_bytes()))


def _stats_with_peak_total_reserved_bytes(stats: MemoryAllocationStats) -> MemoryAllocationStats:
    return replace(
        stats,
        peak_total_reserved_bytes=max(stats.peak_total_reserved_bytes, stats.total_reserved_bytes()),
    )
