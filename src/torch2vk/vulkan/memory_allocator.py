"""Buffer allocation helpers for Vulkan compute execution."""

from __future__ import annotations

from collections.abc import Callable
from vulkan import (
    VK_SHARING_MODE_EXCLUSIVE,
    VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
    VkBufferCreateInfo,
    VkMemoryAllocateInfo,
    VkMemoryPriorityAllocateInfoEXT,
    vkAllocateMemory,
    vkBindBufferMemory,
    vkCreateBuffer,
    vkGetBufferMemoryRequirements,
)
from vulkan._vulkan import ffi, lib as _vulkan_lib

from torch2vk.vulkan.types import tensor_nbytes as tensor_nbytes

from .abi import VkPhysicalDeviceMemoryProperties, memory_requirements
from .buffer import VulkanBuffer


VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_VALUE = 0x00020000
VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_VALUE = 0x00000002
VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_VALUE = 1000060000
_ffi_null = ffi.NULL


def create_buffer(
    *,
    device_handle: object,
    memory_properties: VkPhysicalDeviceMemoryProperties,
    require_device_open: Callable[[], None],
    is_device_closed: Callable[[], bool],
    size: int,
    usage_flags: int,
    memory_property_flags: int,
    memory_priority: float | None = None,
) -> VulkanBuffer:
    require_device_open()
    buffer = vkCreateBuffer(
        device_handle,
        VkBufferCreateInfo(
            size=size,
            usage=usage_flags,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        ),
        None,
    )
    requirements = memory_requirements(vkGetBufferMemoryRequirements(device_handle, buffer))
    memory_type_index = find_memory_type(
        memory_properties,
        requirements.memory_type_bits,
        memory_property_flags,
    )
    allocate_pnext: object | None = None
    if usage_flags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_VALUE:
        allocate_pnext = ffi.new(
            "VkMemoryAllocateFlagsInfo *",
            {
                "sType": VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_VALUE,
                "pNext": _ffi_null,
                "flags": VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_VALUE,
                "deviceMask": 0,
            },
        )
    if memory_priority is not None:
        if memory_priority < 0.0 or memory_priority > 1.0:
            raise ValueError(f"memory_priority must be in [0, 1], got {memory_priority}")
        priority_info = VkMemoryPriorityAllocateInfoEXT(
            sType=VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
            pNext=allocate_pnext,
            priority=float(memory_priority),
        )
        allocate_pnext = ffi.addressof(priority_info)
    memory = vkAllocateMemory(
        device_handle,
        VkMemoryAllocateInfo(
            pNext=allocate_pnext,
            allocationSize=requirements.size,
            memoryTypeIndex=memory_type_index,
        ),
        None,
    )
    vkBindBufferMemory(device_handle, buffer, memory, 0)
    return VulkanBuffer(
        device_handle=device_handle,
        require_device_open=require_device_open,
        is_device_closed=is_device_closed,
        handle=buffer,
        memory=memory,
        size=size,
    )


def set_device_memory_priority(
    *,
    device_handle: object,
    memory: object,
    priority: float,
) -> None:
    if priority < 0.0 or priority > 1.0:
        raise ValueError(f"priority must be in [0, 1], got {priority}")
    vk_get_device_proc_addr = getattr(_vulkan_lib, "vkGetDeviceProcAddr")
    raw = vk_get_device_proc_addr(device_handle, b"vkSetDeviceMemoryPriorityEXT")
    if raw == _ffi_null:
        raise RuntimeError("vkSetDeviceMemoryPriorityEXT is not available on this Vulkan device")
    fn = ffi.cast("PFN_vkSetDeviceMemoryPriorityEXT", raw)
    fn(device_handle, memory, float(priority))


def find_memory_type(
    memory_properties: VkPhysicalDeviceMemoryProperties,
    type_bits: int,
    required_flags: int,
) -> int:
    for index, memory_type in enumerate(memory_properties.memory_types):
        supports_type = type_bits & (1 << index)
        flags = int(memory_type.property_flags)
        if supports_type and (flags & required_flags) == required_flags:
            return index
    raise RuntimeError(f"Could not find Vulkan memory type for bits={type_bits:#x} flags={required_flags:#x}")
