"""Thin Vulkan discovery helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class VulkanPhysicalDevice:
    index: int
    name: str
    compute_queue_family: int | None


@dataclass(slots=True)
class VulkanContext:
    vk: Any
    instance: Any
    physical_device: Any
    device: Any
    compute_queue: Any
    compute_queue_family: int
    physical_device_name: str

    def close(self) -> None:
        self.vk.vkDeviceWaitIdle(self.device)
        self.vk.vkDestroyDevice(self.device, None)
        self.vk.vkDestroyInstance(self.instance, None)

    def create_host_buffer(self, *, nbytes: int) -> VulkanBuffer:
        if nbytes <= 0:
            raise ValueError(f"buffer size must be positive, got {nbytes}")
        buffer_info = self.vk.VkBufferCreateInfo(
            sType=self.vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=nbytes,
            usage=self.vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=self.vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buffer = self.vk.vkCreateBuffer(self.device, buffer_info, None)
        try:
            requirements = self.vk.vkGetBufferMemoryRequirements(self.device, buffer)
            memory_type = _find_memory_type(
                self.vk,
                self.physical_device,
                type_filter=requirements.memoryTypeBits,
                properties=(
                    self.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | self.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                ),
            )
            allocate_info = self.vk.VkMemoryAllocateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=memory_type,
            )
            memory = self.vk.vkAllocateMemory(self.device, allocate_info, None)
            self.vk.vkBindBufferMemory(self.device, buffer, memory, 0)
            return VulkanBuffer(
                context=self,
                buffer=buffer,
                memory=memory,
                nbytes=nbytes,
                allocation_nbytes=int(requirements.size),
            )
        except Exception:
            self.vk.vkDestroyBuffer(self.device, buffer, None)
            raise

    def create_shader_module(self, spirv: bytes) -> VulkanShaderModule:
        if not spirv:
            raise ValueError("SPIR-V bytecode must be non-empty")
        if len(spirv) % 4 != 0:
            raise ValueError(f"SPIR-V bytecode length must be 4-byte aligned, got {len(spirv)}")
        create_info = self.vk.VkShaderModuleCreateInfo(
            sType=self.vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(spirv),
            pCode=spirv,
        )
        module = self.vk.vkCreateShaderModule(self.device, create_info, None)
        return VulkanShaderModule(context=self, module=module)


@dataclass(slots=True)
class VulkanBuffer:
    context: VulkanContext
    buffer: Any
    memory: Any
    nbytes: int
    allocation_nbytes: int

    def close(self) -> None:
        vk = self.context.vk
        vk.vkDestroyBuffer(self.context.device, self.buffer, None)
        vk.vkFreeMemory(self.context.device, self.memory, None)

    def write(self, data: bytes, *, offset: int = 0) -> None:
        if offset < 0 or offset + len(data) > self.nbytes:
            raise ValueError(f"write range [{offset}, {offset + len(data)}) exceeds {self.nbytes}")
        vk = self.context.vk
        mapped = vk.vkMapMemory(self.context.device, self.memory, offset, len(data), 0)
        try:
            mapped[: len(data)] = data
        finally:
            vk.vkUnmapMemory(self.context.device, self.memory)

    def read(self, *, offset: int = 0, nbytes: int | None = None) -> bytes:
        size = self.nbytes - offset if nbytes is None else nbytes
        if offset < 0 or size < 0 or offset + size > self.nbytes:
            raise ValueError(f"read range [{offset}, {offset + size}) exceeds {self.nbytes}")
        vk = self.context.vk
        mapped = vk.vkMapMemory(self.context.device, self.memory, offset, size, 0)
        try:
            return bytes(mapped[:size])
        finally:
            vk.vkUnmapMemory(self.context.device, self.memory)


@dataclass(slots=True)
class VulkanShaderModule:
    context: VulkanContext
    module: Any

    def close(self) -> None:
        self.context.vk.vkDestroyShaderModule(self.context.device, self.module, None)


def enumerate_physical_devices() -> tuple[VulkanPhysicalDevice, ...]:
    vk = importlib.import_module("vulkan")
    instance = _create_instance(vk)
    try:
        devices = cast("list[Any]", vk.vkEnumeratePhysicalDevices(instance))
        infos: list[VulkanPhysicalDevice] = []
        for index, device in enumerate(devices):
            props = vk.vkGetPhysicalDeviceProperties(device)
            infos.append(
                VulkanPhysicalDevice(
                    index=index,
                    name=str(props.deviceName),
                    compute_queue_family=_compute_queue_family(vk, device),
                )
            )
        return tuple(infos)
    finally:
        vk.vkDestroyInstance(instance, None)


def create_compute_context(*, prefer_device_index: int | None = None) -> VulkanContext:
    vk = importlib.import_module("vulkan")
    instance = _create_instance(vk)
    try:
        physical_device, queue_family = _select_physical_device(
            vk,
            instance,
            prefer_device_index=prefer_device_index,
        )
        queue_priority = [1.0]
        queue_create = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=queue_family,
            queueCount=1,
            pQueuePriorities=queue_priority,
        )
        device_create = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create],
        )
        device = vk.vkCreateDevice(physical_device, device_create, None)
        queue = vk.vkGetDeviceQueue(device, queue_family, 0)
        props = vk.vkGetPhysicalDeviceProperties(physical_device)
        return VulkanContext(
            vk=vk,
            instance=instance,
            physical_device=physical_device,
            device=device,
            compute_queue=queue,
            compute_queue_family=queue_family,
            physical_device_name=str(props.deviceName),
        )
    except Exception:
        vk.vkDestroyInstance(instance, None)
        raise


def _create_instance(vk: Any) -> Any:
    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="torch2vk",
        applicationVersion=1,
        pEngineName="torch2vk",
        engineVersion=1,
        apiVersion=vk.VK_MAKE_VERSION(1, 2, 0),
    )
    create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info,
    )
    return vk.vkCreateInstance(create_info, None)


def _select_physical_device(
    vk: Any,
    instance: Any,
    *,
    prefer_device_index: int | None,
) -> tuple[Any, int]:
    devices = cast("list[Any]", vk.vkEnumeratePhysicalDevices(instance))
    if not devices:
        raise RuntimeError("No Vulkan physical devices found")
    if prefer_device_index is not None:
        if prefer_device_index < 0 or prefer_device_index >= len(devices):
            raise ValueError(f"Invalid Vulkan device index {prefer_device_index}")
        queue_family = _compute_queue_family(vk, devices[prefer_device_index])
        if queue_family is None:
            raise RuntimeError(f"Vulkan device {prefer_device_index} has no compute queue")
        return devices[prefer_device_index], queue_family

    for device in devices:
        queue_family = _compute_queue_family(vk, device)
        if queue_family is not None:
            return device, queue_family
    raise RuntimeError("No Vulkan physical device with a compute queue found")


def _compute_queue_family(vk: Any, physical_device: Any) -> int | None:
    queues = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    for index, queue in enumerate(queues):
        if queue.queueCount > 0 and queue.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
            return int(index)
    return None


def _find_memory_type(
    vk: Any,
    physical_device: Any,
    *,
    type_filter: int,
    properties: int,
) -> int:
    memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(physical_device)
    for index in range(memory_properties.memoryTypeCount):
        supported = type_filter & (1 << index)
        memory_type = memory_properties.memoryTypes[index]
        if supported and memory_type.propertyFlags & properties == properties:
            return int(index)
    raise RuntimeError("No compatible Vulkan memory type found")
