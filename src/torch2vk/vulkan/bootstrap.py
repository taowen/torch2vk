"""Vulkan instance/device bootstrap and device enumeration."""

from __future__ import annotations

from dataclasses import dataclass

from vulkan import (
    VK_MAKE_VERSION,
    VK_MEMORY_OVERALLOCATION_BEHAVIOR_ALLOWED_AMD,
    VK_QUEUE_COMPUTE_BIT,
    VK_STRUCTURE_TYPE_DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO_AMD,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    VkPhysicalDevice16BitStorageFeatures,
    VkApplicationInfo,
    VkDeviceCreateInfo,
    VkDeviceMemoryOverallocationCreateInfoAMD,
    VkDeviceQueueCreateInfo,
    VkInstanceCreateInfo,
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR,
    VkPhysicalDeviceFeatures2,
    VkPhysicalDeviceMemoryPriorityFeaturesEXT,
    VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT,
    VkPhysicalDeviceShaderIntegerDotProductFeatures,
    VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR,
    VkPhysicalDeviceSubgroupSizeControlFeatures,
    VkPhysicalDeviceVulkan11Features,
    VkPhysicalDeviceVulkan12Features,
    vkCreateDevice,
    vkCreateInstance,
    vkDestroyInstance,
    vkEnumerateDeviceExtensionProperties,
    vkEnumerateInstanceExtensionProperties,
    vkEnumeratePhysicalDevices,
    vkGetPhysicalDeviceProperties,
    vkGetPhysicalDeviceQueueFamilyProperties,
)
from vulkan._vulkan import ffi

from .abi import device_name_properties
from .capabilities import DeviceFeatureSupport


@dataclass(frozen=True, slots=True)
class PhysicalDeviceInfo:
    index: int
    name: str
    queue_family_index: int
    api_version: int


def _vk_name_to_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError(f"Unexpected Vulkan name value: {value!r}")


def enumerate_compute_devices() -> list[PhysicalDeviceInfo]:
    instance = create_instance()
    try:
        devices: list[PhysicalDeviceInfo] = []
        for index, physical_device in enumerate(vkEnumeratePhysicalDevices(instance)):
            queue_family_index = find_compute_queue_family(physical_device)
            props = device_name_properties(vkGetPhysicalDeviceProperties(physical_device))
            devices.append(
                PhysicalDeviceInfo(
                    index=index,
                    name=_vk_name_to_str(props.device_name),
                    queue_family_index=queue_family_index,
                    api_version=int(props.api_version),
                )
            )
        return devices
    finally:
        vkDestroyInstance(instance, None)


def create_instance() -> object:
    available_extensions = enumerate_instance_extension_names()
    enabled_extensions: list[str] = []
    if "VK_KHR_get_physical_device_properties2" in available_extensions:
        enabled_extensions.append("VK_KHR_get_physical_device_properties2")
    if "VK_EXT_debug_utils" in available_extensions:
        enabled_extensions.append("VK_EXT_debug_utils")
    app_info = VkApplicationInfo(
        pApplicationName="torch2vk",
        applicationVersion=1,
        pEngineName="torch2vk",
        engineVersion=1,
        apiVersion=VK_MAKE_VERSION(1, 3, 0),
    )
    return vkCreateInstance(
        VkInstanceCreateInfo(
            pApplicationInfo=app_info,
            enabledExtensionCount=len(enabled_extensions),
            ppEnabledExtensionNames=enabled_extensions or None,
        ),
        None,
    )


def create_device(
    physical_device: object,
    queue_family_index: int,
    *,
    available_extensions: set[str],
    feature_support: DeviceFeatureSupport,
) -> object:
    queue_infos = [
        VkDeviceQueueCreateInfo(
            queueFamilyIndex=queue_family_index,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
    ]
    extension_names: list[str] = []
    features2 = VkPhysicalDeviceFeatures2(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
    features2.features.shaderInt64 = feature_support.shader_int64
    vulkan11 = VkPhysicalDeviceVulkan11Features(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        storageBuffer16BitAccess=feature_support.storage_buffer_16bit_access,
        uniformAndStorageBuffer16BitAccess=feature_support.uniform_and_storage_buffer_16bit_access,
    )
    vulkan12 = VkPhysicalDeviceVulkan12Features(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        shaderFloat16=feature_support.shader_float16,
        shaderInt8=feature_support.shader_int8,
        bufferDeviceAddress=feature_support.buffer_device_address,
        vulkanMemoryModel=feature_support.vulkan_memory_model,
        vulkanMemoryModelDeviceScope=feature_support.vulkan_memory_model_device_scope,
    )
    storage_16bit = VkPhysicalDevice16BitStorageFeatures(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        storageBuffer16BitAccess=feature_support.storage_buffer_16bit_access,
        uniformAndStorageBuffer16BitAccess=feature_support.uniform_and_storage_buffer_16bit_access,
    )
    cooperative_matrix = VkPhysicalDeviceCooperativeMatrixFeaturesKHR(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        cooperativeMatrix=feature_support.cooperative_matrix,
        cooperativeMatrixRobustBufferAccess=feature_support.cooperative_matrix_robust_buffer_access,
    )
    subgroup_size_control = VkPhysicalDeviceSubgroupSizeControlFeatures(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
        subgroupSizeControl=feature_support.subgroup_size_control,
        computeFullSubgroups=feature_support.compute_full_subgroups,
    )
    shader_integer_dot_product = VkPhysicalDeviceShaderIntegerDotProductFeatures(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
        shaderIntegerDotProduct=feature_support.shader_integer_dot_product,
    )
    subgroup_uniform_control_flow = VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR,
        shaderSubgroupUniformControlFlow=feature_support.shader_subgroup_uniform_control_flow,
    )
    memory_priority = VkPhysicalDeviceMemoryPriorityFeaturesEXT(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
        memoryPriority=feature_support.memory_priority,
    )
    pageable_device_local_memory = VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT,
        pageableDeviceLocalMemory=feature_support.pageable_device_local_memory,
    )
    features2.pNext = ffi.addressof(vulkan11)
    vulkan11.pNext = ffi.addressof(vulkan12)
    vulkan12.pNext = ffi.addressof(storage_16bit)
    storage_16bit.pNext = ffi.addressof(cooperative_matrix)
    cooperative_matrix.pNext = ffi.addressof(subgroup_size_control)
    subgroup_size_control.pNext = ffi.addressof(shader_integer_dot_product)
    shader_integer_dot_product.pNext = ffi.addressof(subgroup_uniform_control_flow)
    subgroup_uniform_control_flow.pNext = ffi.addressof(memory_priority)
    memory_priority.pNext = ffi.addressof(pageable_device_local_memory)

    if "VK_KHR_16bit_storage" in available_extensions and feature_support.storage_buffer_16bit_access:
        extension_names.append("VK_KHR_16bit_storage")
    if "VK_KHR_buffer_device_address" in available_extensions and feature_support.buffer_device_address:
        extension_names.append("VK_KHR_buffer_device_address")
    if "VK_KHR_cooperative_matrix" in available_extensions and feature_support.cooperative_matrix:
        extension_names.append("VK_KHR_cooperative_matrix")
    if "VK_KHR_shader_subgroup_uniform_control_flow" in available_extensions and feature_support.shader_subgroup_uniform_control_flow:
        extension_names.append("VK_KHR_shader_subgroup_uniform_control_flow")
    if "VK_KHR_vulkan_memory_model" in available_extensions and feature_support.vulkan_memory_model:
        extension_names.append("VK_KHR_vulkan_memory_model")
    if "VK_KHR_shader_float16_int8" in available_extensions and (feature_support.shader_float16 or feature_support.shader_int8):
        extension_names.append("VK_KHR_shader_float16_int8")
    if "VK_KHR_shader_integer_dot_product" in available_extensions and feature_support.shader_integer_dot_product:
        extension_names.append("VK_KHR_shader_integer_dot_product")
    if "VK_EXT_subgroup_size_control" in available_extensions and feature_support.subgroup_size_control:
        extension_names.append("VK_EXT_subgroup_size_control")
    if "VK_EXT_memory_priority" in available_extensions and feature_support.memory_priority:
        extension_names.append("VK_EXT_memory_priority")
    if "VK_EXT_pageable_device_local_memory" in available_extensions and feature_support.pageable_device_local_memory:
        extension_names.append("VK_EXT_pageable_device_local_memory")
    device_create_pnext = ffi.addressof(features2)
    if "VK_AMD_memory_overallocation_behavior" in available_extensions:
        extension_names.append("VK_AMD_memory_overallocation_behavior")
        memory_overallocation = VkDeviceMemoryOverallocationCreateInfoAMD(
            sType=VK_STRUCTURE_TYPE_DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO_AMD,
            pNext=device_create_pnext,
            overallocationBehavior=VK_MEMORY_OVERALLOCATION_BEHAVIOR_ALLOWED_AMD,
        )
        device_create_pnext = ffi.addressof(memory_overallocation)

    device_create_info = VkDeviceCreateInfo(
        queueCreateInfoCount=1,
        pQueueCreateInfos=queue_infos,
        enabledExtensionCount=len(extension_names),
        ppEnabledExtensionNames=extension_names or None,
        pNext=device_create_pnext,
    )
    return vkCreateDevice(physical_device, device_create_info, None)


def find_compute_queue_family(physical_device: object) -> int:
    queue_properties = vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    for index, props in enumerate(queue_properties):
        if props.queueFlags & VK_QUEUE_COMPUTE_BIT:
            return index
    raise RuntimeError("No compute-capable queue family found")


def enumerate_device_extension_names(physical_device: object) -> set[str]:
    return {
        _vk_name_to_str(extension.extensionName)
        for extension in vkEnumerateDeviceExtensionProperties(physical_device, None)
    }


def enumerate_instance_extension_names() -> set[str]:
    return {
        _vk_name_to_str(extension.extensionName)
        for extension in vkEnumerateInstanceExtensionProperties(None)
    }
