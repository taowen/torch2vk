"""Thin Vulkan discovery helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class VulkanPhysicalDevice:
    index: int
    name: str


def enumerate_physical_devices() -> tuple[VulkanPhysicalDevice, ...]:
    vk = importlib.import_module("vulkan")
    instance = _create_instance(vk)
    try:
        devices = cast("list[Any]", vk.vkEnumeratePhysicalDevices(instance))
        infos: list[VulkanPhysicalDevice] = []
        for index, device in enumerate(devices):
            props = vk.vkGetPhysicalDeviceProperties(device)
            infos.append(VulkanPhysicalDevice(index=index, name=str(props.deviceName)))
        return tuple(infos)
    finally:
        vk.vkDestroyInstance(instance, None)


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
