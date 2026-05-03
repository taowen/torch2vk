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
