"""Buffer allocation helpers for Vulkan compute execution."""

from __future__ import annotations

from collections.abc import Callable
from vulkan import (
    VK_SHARING_MODE_EXCLUSIVE,
    VkBufferCreateInfo,
    VkMemoryAllocateInfo,
    vkAllocateMemory,
    vkBindBufferMemory,
    vkCreateBuffer,
    vkGetBufferMemoryRequirements,
)
from vulkan._vulkan import ffi

from torch2vk.vulkan.types import TensorSpec, concrete_nbytes

from .abi import VkPhysicalDeviceMemoryProperties, memory_requirements
from .buffer import VulkanBuffer


VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_VALUE = 0x00020000
VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_VALUE = 0x00000002
VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_VALUE = 1000060000
_ffi_null = ffi.NULL


def tensor_nbytes(spec: TensorSpec) -> int:
    concrete_shape: list[int] = []
    for dim in spec.shape:
        if not isinstance(dim, int):
            raise ValueError(f"Expected concrete tensor shape, got {spec.shape}")
        concrete_shape.append(dim)
    return concrete_nbytes(dtype=spec.dtype, shape=tuple(concrete_shape))


def create_buffer(
    *,
    device_handle: object,
    memory_properties: VkPhysicalDeviceMemoryProperties,
    require_device_open: Callable[[], None],
    is_device_closed: Callable[[], bool],
    size: int,
    usage_flags: int,
    memory_property_flags: int,
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
