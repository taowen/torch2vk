"""Vulkan-backed device buffer with explicit map/unmap helpers."""

from __future__ import annotations

import struct
import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Protocol, TypeAlias, runtime_checkable

from vulkan import VK_NULL_HANDLE, vkDestroyBuffer, vkFreeMemory, vkMapMemory, vkUnmapMemory
from vulkan._vulkan import ffi

BufferData: TypeAlias = bytes | bytearray | memoryview


class _GetBufferDeviceAddress(Protocol):
    def __call__(self, device: object, info: object) -> int: ...


@runtime_checkable
class _MappedMemory(Protocol):
    def __setitem__(self, key: slice, value: BufferData) -> None: ...

    def __getitem__(self, key: slice) -> bytes: ...


_vulkan_cffi = importlib.import_module("vulkan._vulkan")
_vulkan_lib = getattr(_vulkan_cffi, "lib")
_ffi_null = ffi.NULL


def _get_buffer_device_address_proc() -> _GetBufferDeviceAddress:
    fn = getattr(_vulkan_lib, "vkGetBufferDeviceAddress")
    if not callable(fn):
        raise TypeError("vkGetBufferDeviceAddress is not callable")

    def get_buffer_device_address(device: object, info: object) -> int:
        result = fn(device, info)
        if not isinstance(result, int):
            raise TypeError(f"vkGetBufferDeviceAddress returned {type(result).__name__}")
        return int(result)

    return get_buffer_device_address


_vk_get_buffer_device_address = _get_buffer_device_address_proc()


def _mapped_memory(value: object) -> _MappedMemory:
    if not isinstance(value, _MappedMemory):
        raise TypeError(f"vkMapMemory returned non-buffer-like object {type(value).__name__}")
    return value


@dataclass(slots=True)
class VulkanBuffer:
    device_handle: object
    require_device_open: Callable[[], None]
    is_device_closed: Callable[[], bool]
    handle: object
    memory: object
    size: int
    _mapped_memory: _MappedMemory | None = field(default=None, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return self.handle == VK_NULL_HANDLE or self.memory == VK_NULL_HANDLE

    def close(self) -> None:
        if self.closed:
            return
        if self.is_device_closed():
            self.handle = VK_NULL_HANDLE
            self.memory = VK_NULL_HANDLE
            self._mapped_memory = None
            return
        self.unmap_persistent()
        if self.handle != VK_NULL_HANDLE:
            vkDestroyBuffer(self.device_handle, self.handle, None)
            self.handle = VK_NULL_HANDLE
        if self.memory != VK_NULL_HANDLE:
            vkFreeMemory(self.device_handle, self.memory, None)
            self.memory = VK_NULL_HANDLE

    def __enter__(self) -> "VulkanBuffer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def write_bytes(self, data: BufferData) -> None:
        self.write_bytes_at(0, data)

    def map_persistent(self) -> None:
        self.require_device_open()
        if self.closed:
            raise RuntimeError("VulkanBuffer is closed")
        if self._mapped_memory is None:
            self._mapped_memory = _mapped_memory(vkMapMemory(self.device_handle, self.memory, 0, self.size, 0))

    def unmap_persistent(self) -> None:
        if self._mapped_memory is None:
            return
        vkUnmapMemory(self.device_handle, self.memory)
        self._mapped_memory = None

    def write_bytes_at(self, offset: int, data: BufferData) -> None:
        self.require_device_open()
        if self.closed:
            raise RuntimeError("VulkanBuffer is closed")
        if offset < 0:
            raise ValueError(f"buffer write offset must be non-negative, got {offset}")
        if offset + len(data) > self.size:
            raise ValueError(
                f"buffer size {self.size} is smaller than write ending at {offset + len(data)} bytes"
            )
        mapped = self._mapped_memory
        if mapped is not None:
            mapped[offset : offset + len(data)] = data
            return
        transient = _mapped_memory(vkMapMemory(self.device_handle, self.memory, 0, self.size, 0))
        try:
            transient[offset : offset + len(data)] = data
        finally:
            vkUnmapMemory(self.device_handle, self.memory)

    def read_bytes(self) -> bytes:
        return self.read_bytes_at(0, self.size)

    def read_bytes_at(self, offset: int, size: int) -> bytes:
        self.require_device_open()
        if self.closed:
            raise RuntimeError("VulkanBuffer is closed")
        if offset < 0 or size < 0:
            raise ValueError(f"buffer read requires non-negative offset and size, got offset={offset}, size={size}")
        if offset + size > self.size:
            raise ValueError(
                f"buffer size {self.size} is smaller than read ending at {offset + size} bytes"
            )
        mapped = self._mapped_memory
        if mapped is not None:
            return bytes(mapped[offset : offset + size])
        transient = _mapped_memory(vkMapMemory(self.device_handle, self.memory, 0, self.size, 0))
        try:
            return bytes(transient[offset : offset + size])
        finally:
            vkUnmapMemory(self.device_handle, self.memory)

    def write_f32(self, values: list[float] | tuple[float, ...]) -> None:
        self.write_bytes(struct.pack(f"{len(values)}f", *values))

    def read_f32(self, count: int) -> list[float]:
        data = self.read_bytes()[: count * 4]
        return list(struct.unpack(f"{count}f", data))

    @property
    def device_address(self) -> int:
        self.require_device_open()
        if self.closed:
            raise RuntimeError("VulkanBuffer is closed")
        info = ffi.new(
            "VkBufferDeviceAddressInfo *",
            {
                "sType": 1000244001,
                "pNext": _ffi_null,
                "buffer": self.handle,
            },
        )
        return int(_vk_get_buffer_device_address(self.device_handle, info))
