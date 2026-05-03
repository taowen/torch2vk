"""Physical-device capability queries and shader execution requirement checks."""

from __future__ import annotations

from dataclasses import dataclass

from vulkan import (
    VK_COMPONENT_TYPE_FLOAT16_KHR,
    VK_COMPONENT_TYPE_FLOAT32_KHR,
    VK_COMPONENT_TYPE_SINT16_KHR,
    VK_COMPONENT_TYPE_SINT32_KHR,
    VK_COMPONENT_TYPE_SINT8_KHR,
    VK_COMPONENT_TYPE_UINT16_KHR,
    VK_COMPONENT_TYPE_UINT32_KHR,
    VK_COMPONENT_TYPE_UINT8_KHR,
    VK_SCOPE_SUBGROUP_KHR,
    VK_SHADER_STAGE_COMPUTE_BIT,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES,
    VkPhysicalDevice16BitStorageFeatures,
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR,
    VkPhysicalDeviceFeatures2,
    VkPhysicalDeviceProperties2,
    VkPhysicalDeviceShaderIntegerDotProductFeatures,
    VkPhysicalDeviceShaderIntegerDotProductProperties,
    VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR,
    VkPhysicalDeviceSubgroupProperties,
    VkPhysicalDeviceSubgroupSizeControlFeatures,
    VkPhysicalDeviceSubgroupSizeControlProperties,
    VkPhysicalDeviceVulkan11Features,
    VkPhysicalDeviceVulkan12Features,
    VkPhysicalDeviceVulkan13Properties,
    vkGetInstanceProcAddr,
    vkGetPhysicalDeviceFeatures2,
    vkGetPhysicalDeviceProperties2,
)
from vulkan._vulkan import ffi

from .abi import cooperative_matrix_properties_proc
from .shader_execution_requirements import (
    CooperativeMatrixRequirements,
    ShaderComponentTypeName,
    ShaderExecutionRequirements,
    ShaderScopeName,
    SubgroupRequirements,
)


@dataclass(frozen=True, slots=True)
class CooperativeMatrixSupport:
    m_size: int
    n_size: int
    k_size: int
    a_type: int
    b_type: int
    c_type: int
    result_type: int
    scope: int
    saturating_accumulation: bool


@dataclass(frozen=True, slots=True)
class DeviceFeatureSupport:
    cooperative_matrix: bool
    cooperative_matrix_robust_buffer_access: bool
    storage_buffer_16bit_access: bool
    uniform_and_storage_buffer_16bit_access: bool
    shader_float16: bool
    shader_int8: bool
    shader_int64: bool
    shader_integer_dot_product: bool
    buffer_device_address: bool
    shader_subgroup_uniform_control_flow: bool
    subgroup_size_control: bool
    compute_full_subgroups: bool
    vulkan_memory_model: bool
    vulkan_memory_model_device_scope: bool


@dataclass(frozen=True, slots=True)
class SubgroupSizeControlSupport:
    subgroup_size: int
    min_subgroup_size: int
    max_subgroup_size: int
    required_subgroup_size_stages: int


@dataclass(frozen=True, slots=True)
class IntegerDotProductSupport:
    packed_4x8_signed_accelerated: bool


_COMPONENT_TYPE_BY_NAME: dict[ShaderComponentTypeName, int] = {
    "float16": VK_COMPONENT_TYPE_FLOAT16_KHR,
    "float32": VK_COMPONENT_TYPE_FLOAT32_KHR,
    "sint8": VK_COMPONENT_TYPE_SINT8_KHR,
    "uint8": VK_COMPONENT_TYPE_UINT8_KHR,
    "sint16": VK_COMPONENT_TYPE_SINT16_KHR,
    "uint16": VK_COMPONENT_TYPE_UINT16_KHR,
    "sint32": VK_COMPONENT_TYPE_SINT32_KHR,
    "uint32": VK_COMPONENT_TYPE_UINT32_KHR,
}

_SCOPE_BY_NAME: dict[ShaderScopeName, int] = {
    "subgroup": VK_SCOPE_SUBGROUP_KHR,
}


def supports_shader_execution_requirements(
    *,
    feature_support: DeviceFeatureSupport,
    subgroup_size_control_support: SubgroupSizeControlSupport,
    cooperative_matrix_support: tuple[CooperativeMatrixSupport, ...],
    execution_requirements: ShaderExecutionRequirements | None,
) -> bool:
    try:
        require_shader_execution_requirements(
            feature_support=feature_support,
            subgroup_size_control_support=subgroup_size_control_support,
            cooperative_matrix_support=cooperative_matrix_support,
            execution_requirements=execution_requirements,
        )
    except RuntimeError:
        return False
    return True


def require_shader_execution_requirements(
    *,
    feature_support: DeviceFeatureSupport,
    subgroup_size_control_support: SubgroupSizeControlSupport,
    cooperative_matrix_support: tuple[CooperativeMatrixSupport, ...],
    execution_requirements: ShaderExecutionRequirements | None,
) -> None:
    if execution_requirements is None:
        return
    if execution_requirements.subgroup is not None:
        require_subgroup_requirements(
            feature_support=feature_support,
            subgroup_size_control_support=subgroup_size_control_support,
            subgroup=execution_requirements.subgroup,
        )
    if execution_requirements.cooperative_matrix is not None:
        require_cooperative_matrix_requirements(
            cooperative_matrix_support=cooperative_matrix_support,
            cooperative_matrix=execution_requirements.cooperative_matrix,
        )
    if execution_requirements.require_storage_buffer_16bit_access and not feature_support.storage_buffer_16bit_access:
        raise RuntimeError("This runtime requires storageBuffer16BitAccess support")
    if execution_requirements.require_shader_int64 and not feature_support.shader_int64:
        raise RuntimeError("This runtime requires shaderInt64 support")
    if execution_requirements.require_buffer_device_address and not feature_support.buffer_device_address:
        raise RuntimeError("This runtime requires bufferDeviceAddress support")
    if execution_requirements.require_integer_dot_product and not feature_support.shader_integer_dot_product:
        raise RuntimeError("This runtime requires VK_KHR_shader_integer_dot_product support")


def require_subgroup_requirements(
    *,
    feature_support: DeviceFeatureSupport,
    subgroup_size_control_support: SubgroupSizeControlSupport,
    subgroup: SubgroupRequirements,
) -> None:
    subgroup_size = subgroup.required_size
    if not feature_support.subgroup_size_control:
        raise RuntimeError(f"This runtime requires subgroupSizeControl to request subgroup size {subgroup_size}")
    if subgroup.require_full_subgroups and not feature_support.compute_full_subgroups:
        raise RuntimeError("This runtime requires computeFullSubgroups to use REQUIRE_FULL_SUBGROUPS compute pipelines")
    if not (subgroup_size_control_support.required_subgroup_size_stages & VK_SHADER_STAGE_COMPUTE_BIT):
        raise RuntimeError("This runtime does not allow required subgroup sizes in compute pipelines")
    min_subgroup_size = subgroup_size_control_support.min_subgroup_size
    max_subgroup_size = subgroup_size_control_support.max_subgroup_size
    if subgroup_size < min_subgroup_size or subgroup_size > max_subgroup_size:
        raise RuntimeError(
            f"Requested subgroup size {subgroup_size} is outside supported range [{min_subgroup_size}, {max_subgroup_size}]"
        )


def require_cooperative_matrix_requirements(
    *,
    cooperative_matrix_support: tuple[CooperativeMatrixSupport, ...],
    cooperative_matrix: CooperativeMatrixRequirements,
) -> None:
    requested_scope = _SCOPE_BY_NAME[cooperative_matrix.scope]
    requested_a_type = _COMPONENT_TYPE_BY_NAME[cooperative_matrix.a_type]
    requested_b_type = _COMPONENT_TYPE_BY_NAME[cooperative_matrix.b_type]
    requested_c_type = _COMPONENT_TYPE_BY_NAME[cooperative_matrix.c_type]
    requested_result_type = _COMPONENT_TYPE_BY_NAME[cooperative_matrix.result_type]
    for support in cooperative_matrix_support:
        if (
            support.m_size == cooperative_matrix.m_size
            and support.n_size == cooperative_matrix.n_size
            and support.k_size == cooperative_matrix.k_size
            and support.a_type == requested_a_type
            and support.b_type == requested_b_type
            and support.c_type == requested_c_type
            and support.result_type == requested_result_type
            and support.scope == requested_scope
            and support.saturating_accumulation == cooperative_matrix.saturating_accumulation
        ):
            return
    raise RuntimeError(
        "This runtime requires VK_KHR_cooperative_matrix support for "
        f"{cooperative_matrix.scope}-scope "
        f"{cooperative_matrix.m_size}x{cooperative_matrix.n_size}x{cooperative_matrix.k_size} "
        f"{cooperative_matrix.a_type} x {cooperative_matrix.b_type} -> "
        f"{cooperative_matrix.result_type} accumulation"
    )


def query_device_feature_support(physical_device: object) -> DeviceFeatureSupport:
    features2 = VkPhysicalDeviceFeatures2(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
    vulkan11 = VkPhysicalDeviceVulkan11Features(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
    vulkan12 = VkPhysicalDeviceVulkan12Features(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
    storage_16bit = VkPhysicalDevice16BitStorageFeatures(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES)
    cooperative_matrix = VkPhysicalDeviceCooperativeMatrixFeaturesKHR(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
    )
    subgroup_size_control = VkPhysicalDeviceSubgroupSizeControlFeatures(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES
    )
    shader_integer_dot_product = VkPhysicalDeviceShaderIntegerDotProductFeatures(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES
    )
    subgroup_uniform_control_flow = VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR
    )
    features2.pNext = ffi.addressof(vulkan11)
    vulkan11.pNext = ffi.addressof(vulkan12)
    vulkan12.pNext = ffi.addressof(storage_16bit)
    storage_16bit.pNext = ffi.addressof(cooperative_matrix)
    cooperative_matrix.pNext = ffi.addressof(subgroup_size_control)
    subgroup_size_control.pNext = ffi.addressof(shader_integer_dot_product)
    shader_integer_dot_product.pNext = ffi.addressof(subgroup_uniform_control_flow)
    vkGetPhysicalDeviceFeatures2(physical_device, features2)
    return DeviceFeatureSupport(
        cooperative_matrix=bool(cooperative_matrix.cooperativeMatrix),
        cooperative_matrix_robust_buffer_access=bool(cooperative_matrix.cooperativeMatrixRobustBufferAccess),
        storage_buffer_16bit_access=bool(vulkan11.storageBuffer16BitAccess or storage_16bit.storageBuffer16BitAccess),
        uniform_and_storage_buffer_16bit_access=bool(
            vulkan11.uniformAndStorageBuffer16BitAccess or storage_16bit.uniformAndStorageBuffer16BitAccess
        ),
        shader_float16=bool(vulkan12.shaderFloat16),
        shader_int8=bool(vulkan12.shaderInt8),
        shader_int64=bool(features2.features.shaderInt64),
        shader_integer_dot_product=bool(shader_integer_dot_product.shaderIntegerDotProduct),
        buffer_device_address=bool(vulkan12.bufferDeviceAddress),
        shader_subgroup_uniform_control_flow=bool(subgroup_uniform_control_flow.shaderSubgroupUniformControlFlow),
        subgroup_size_control=bool(subgroup_size_control.subgroupSizeControl),
        compute_full_subgroups=bool(subgroup_size_control.computeFullSubgroups),
        vulkan_memory_model=bool(vulkan12.vulkanMemoryModel),
        vulkan_memory_model_device_scope=bool(vulkan12.vulkanMemoryModelDeviceScope),
    )


def query_subgroup_size_control_support(physical_device: object) -> SubgroupSizeControlSupport:
    properties2 = VkPhysicalDeviceProperties2(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2)
    subgroup_properties = VkPhysicalDeviceSubgroupProperties(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES)
    subgroup_size_control_properties = VkPhysicalDeviceSubgroupSizeControlProperties(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES
    )
    vulkan13_properties = VkPhysicalDeviceVulkan13Properties(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES)
    properties2.pNext = ffi.addressof(subgroup_properties)
    subgroup_properties.pNext = ffi.addressof(subgroup_size_control_properties)
    subgroup_size_control_properties.pNext = ffi.addressof(vulkan13_properties)
    vkGetPhysicalDeviceProperties2(physical_device, properties2)
    required_subgroup_size_stages = int(vulkan13_properties.requiredSubgroupSizeStages)
    if required_subgroup_size_stages == 0:
        required_subgroup_size_stages = int(subgroup_size_control_properties.requiredSubgroupSizeStages)
    return SubgroupSizeControlSupport(
        subgroup_size=int(subgroup_properties.subgroupSize),
        min_subgroup_size=int(vulkan13_properties.minSubgroupSize or subgroup_size_control_properties.minSubgroupSize),
        max_subgroup_size=int(vulkan13_properties.maxSubgroupSize or subgroup_size_control_properties.maxSubgroupSize),
        required_subgroup_size_stages=required_subgroup_size_stages,
    )


def query_integer_dot_product_support(physical_device: object) -> IntegerDotProductSupport:
    properties2 = VkPhysicalDeviceProperties2(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2)
    integer_dot_product_properties = VkPhysicalDeviceShaderIntegerDotProductProperties(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES
    )
    properties2.pNext = ffi.addressof(integer_dot_product_properties)
    vkGetPhysicalDeviceProperties2(physical_device, properties2)
    return IntegerDotProductSupport(
        packed_4x8_signed_accelerated=bool(
            integer_dot_product_properties.integerDotProduct4x8BitPackedSignedAccelerated
        ),
    )


def query_cooperative_matrix_support(
    *,
    instance: object,
    physical_device: object,
    available_extensions: set[str],
) -> tuple[CooperativeMatrixSupport, ...]:
    if "VK_KHR_cooperative_matrix" not in available_extensions:
        return ()
    query_fn = cooperative_matrix_properties_proc(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")
    )
    if query_fn is None:
        return ()
    try:
        properties = query_fn(physical_device)
    except TypeError:
        return ()
    return tuple(
        CooperativeMatrixSupport(
            m_size=prop.m_size,
            n_size=prop.n_size,
            k_size=prop.k_size,
            a_type=prop.a_type,
            b_type=prop.b_type,
            c_type=prop.c_type,
            result_type=prop.result_type,
            scope=int(prop.scope),
            saturating_accumulation=prop.saturating_accumulation,
        )
        for prop in properties
    )
