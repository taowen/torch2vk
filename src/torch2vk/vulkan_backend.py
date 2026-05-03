"""Thin Vulkan discovery helpers."""

from __future__ import annotations

import importlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from .shader import ShaderContract


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

    def create_descriptor_set_layout(self, contract: ShaderContract) -> VulkanDescriptorSetLayout:
        bindings = [
            _descriptor_layout_binding(
                self.vk,
                binding=binding.binding,
                descriptor_type=binding.descriptor_type,
            )
            for binding in contract.bindings
        ]
        bindings.extend(

                _descriptor_layout_binding(
                    self.vk,
                    binding=resource.binding,
                    descriptor_type=resource.descriptor_type,
                )
                for resource in contract.resources

        )
        bindings.extend(

                _descriptor_layout_binding(
                    self.vk,
                    binding=uniform.binding,
                    descriptor_type="uniform_buffer",
                )
                for uniform in contract.uniforms

        )
        create_info = self.vk.VkDescriptorSetLayoutCreateInfo(
            sType=self.vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        layout = self.vk.vkCreateDescriptorSetLayout(self.device, create_info, None)
        return VulkanDescriptorSetLayout(context=self, layout=layout)

    def create_descriptor_pool(
        self,
        contract: ShaderContract,
        *,
        max_sets: int = 1,
    ) -> VulkanDescriptorPool:
        if max_sets <= 0:
            raise ValueError(f"max_sets must be positive, got {max_sets}")
        counts = _descriptor_type_counts(self.vk, contract)
        pool_sizes = [
            self.vk.VkDescriptorPoolSize(
                type=descriptor_type,
                descriptorCount=count * max_sets,
            )
            for descriptor_type, count in sorted(counts.items())
        ]
        create_info = self.vk.VkDescriptorPoolCreateInfo(
            sType=self.vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=max_sets,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        pool = self.vk.vkCreateDescriptorPool(self.device, create_info, None)
        return VulkanDescriptorPool(context=self, pool=pool)

    def allocate_descriptor_set(
        self,
        *,
        descriptor_pool: VulkanDescriptorPool,
        descriptor_set_layout: VulkanDescriptorSetLayout,
    ) -> VulkanDescriptorSet:
        allocate_info = self.vk.VkDescriptorSetAllocateInfo(
            sType=self.vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptor_pool.pool,
            descriptorSetCount=1,
            pSetLayouts=[descriptor_set_layout.layout],
        )
        descriptor_set = self.vk.vkAllocateDescriptorSets(self.device, allocate_info)[0]
        return VulkanDescriptorSet(context=self, descriptor_set=descriptor_set)

    def update_descriptor_set(
        self,
        descriptor_set: VulkanDescriptorSet,
        descriptors: Mapping[int, VulkanBuffer],
        *,
        descriptor_types: Mapping[int, str],
    ) -> None:
        writes: list[Any] = []
        buffer_infos: list[Any] = []
        for binding, buffer in sorted(descriptors.items()):
            descriptor_type = descriptor_types[binding]
            buffer_info = self.vk.VkDescriptorBufferInfo(
                buffer=buffer.buffer,
                offset=0,
                range=buffer.nbytes,
            )
            buffer_infos.append(buffer_info)
            writes.append(
                self.vk.VkWriteDescriptorSet(
                    sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set.descriptor_set,
                    dstBinding=binding,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=_descriptor_type(self.vk, descriptor_type),
                    pBufferInfo=[buffer_info],
                )
            )
        self.vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

    def create_pipeline_layout(
        self,
        contract: ShaderContract,
        descriptor_set_layout: VulkanDescriptorSetLayout,
    ) -> VulkanPipelineLayout:
        push_ranges: list[Any] = []
        if contract.push_constants is not None:
            push_ranges.append(
                self.vk.VkPushConstantRange(
                    stageFlags=self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    offset=0,
                    size=contract.push_constants.size,
                )
            )
        create_info = self.vk.VkPipelineLayoutCreateInfo(
            sType=self.vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[descriptor_set_layout.layout],
            pushConstantRangeCount=len(push_ranges),
            pPushConstantRanges=push_ranges,
        )
        layout = self.vk.vkCreatePipelineLayout(self.device, create_info, None)
        return VulkanPipelineLayout(context=self, layout=layout)

    def create_compute_pipeline(
        self,
        *,
        shader_module: VulkanShaderModule,
        pipeline_layout: VulkanPipelineLayout,
        specialization_constants: Mapping[int, int] | None = None,
    ) -> VulkanComputePipeline:
        specialization_info = None
        if specialization_constants:
            specialization_info = _specialization_info(self.vk, specialization_constants)
        stage = self.vk.VkPipelineShaderStageCreateInfo(
            sType=self.vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module.module,
            pName="main",
            pSpecializationInfo=specialization_info,
        )
        create_info = self.vk.VkComputePipelineCreateInfo(
            sType=self.vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage,
            layout=pipeline_layout.layout,
        )
        pipeline = self.vk.vkCreateComputePipelines(
            self.device,
            self.vk.VK_NULL_HANDLE,
            1,
            [create_info],
            None,
        )[0]
        return VulkanComputePipeline(context=self, pipeline=pipeline)

    def create_command_pool(self) -> VulkanCommandPool:
        create_info = self.vk.VkCommandPoolCreateInfo(
            sType=self.vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.compute_queue_family,
            flags=self.vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        pool = self.vk.vkCreateCommandPool(self.device, create_info, None)
        return VulkanCommandPool(context=self, pool=pool)

    def create_fence(self) -> VulkanFence:
        create_info = self.vk.VkFenceCreateInfo(sType=self.vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        fence = self.vk.vkCreateFence(self.device, create_info, None)
        return VulkanFence(context=self, fence=fence)


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


@dataclass(slots=True)
class VulkanDescriptorSetLayout:
    context: VulkanContext
    layout: Any

    def close(self) -> None:
        self.context.vk.vkDestroyDescriptorSetLayout(self.context.device, self.layout, None)


@dataclass(slots=True)
class VulkanDescriptorPool:
    context: VulkanContext
    pool: Any

    def close(self) -> None:
        self.context.vk.vkDestroyDescriptorPool(self.context.device, self.pool, None)


@dataclass(slots=True)
class VulkanDescriptorSet:
    context: VulkanContext
    descriptor_set: Any


@dataclass(slots=True)
class VulkanPipelineLayout:
    context: VulkanContext
    layout: Any

    def close(self) -> None:
        self.context.vk.vkDestroyPipelineLayout(self.context.device, self.layout, None)


@dataclass(slots=True)
class VulkanComputePipeline:
    context: VulkanContext
    pipeline: Any

    def close(self) -> None:
        self.context.vk.vkDestroyPipeline(self.context.device, self.pipeline, None)


@dataclass(slots=True)
class VulkanCommandPool:
    context: VulkanContext
    pool: Any

    def close(self) -> None:
        self.context.vk.vkDestroyCommandPool(self.context.device, self.pool, None)

    def allocate_command_buffer(self) -> VulkanCommandBuffer:
        vk = self.context.vk
        allocate_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        command_buffer = vk.vkAllocateCommandBuffers(self.context.device, allocate_info)[0]
        return VulkanCommandBuffer(context=self.context, command_buffer=command_buffer)


@dataclass(slots=True)
class VulkanCommandBuffer:
    context: VulkanContext
    command_buffer: Any

    def begin(self) -> None:
        begin_info = self.context.vk.VkCommandBufferBeginInfo(
            sType=self.context.vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=self.context.vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        self.context.vk.vkBeginCommandBuffer(self.command_buffer, begin_info)

    def end(self) -> None:
        self.context.vk.vkEndCommandBuffer(self.command_buffer)

    def bind_compute_pipeline(self, pipeline: VulkanComputePipeline) -> None:
        self.context.vk.vkCmdBindPipeline(
            self.command_buffer,
            self.context.vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline,
        )

    def bind_descriptor_set(
        self,
        *,
        pipeline_layout: VulkanPipelineLayout,
        descriptor_set: VulkanDescriptorSet,
    ) -> None:
        self.context.vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            self.context.vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout.layout,
            0,
            1,
            [descriptor_set.descriptor_set],
            0,
            None,
        )

    def dispatch(self, x: int, y: int = 1, z: int = 1) -> None:
        if x <= 0 or y <= 0 or z <= 0:
            raise ValueError(f"dispatch dimensions must be positive, got {(x, y, z)}")
        self.context.vk.vkCmdDispatch(self.command_buffer, x, y, z)

    def submit_and_wait(self, fence: VulkanFence) -> None:
        submit_info = self.context.vk.VkSubmitInfo(
            sType=self.context.vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffer],
        )
        self.context.vk.vkQueueSubmit(self.context.compute_queue, 1, [submit_info], fence.fence)
        self.context.vk.vkWaitForFences(
            self.context.device,
            1,
            [fence.fence],
            self.context.vk.VK_TRUE,
            10_000_000_000,
        )


@dataclass(slots=True)
class VulkanFence:
    context: VulkanContext
    fence: Any

    def close(self) -> None:
        self.context.vk.vkDestroyFence(self.context.device, self.fence, None)


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


def _descriptor_layout_binding(
    vk: Any,
    *,
    binding: int,
    descriptor_type: str,
) -> Any:
    return vk.VkDescriptorSetLayoutBinding(
        binding=binding,
        descriptorType=_descriptor_type(vk, descriptor_type),
        descriptorCount=1,
        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
    )


def _descriptor_type(vk: Any, descriptor_type: str) -> int:
    if descriptor_type == "storage_buffer":
        return int(vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
    if descriptor_type == "uniform_buffer":
        return int(vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
    raise ValueError(f"Unsupported descriptor type {descriptor_type!r}")


def _descriptor_type_counts(vk: Any, contract: ShaderContract) -> dict[int, int]:
    counts: dict[int, int] = {}
    for binding in contract.bindings:
        descriptor_type = _descriptor_type(vk, binding.descriptor_type)
        counts[descriptor_type] = counts.get(descriptor_type, 0) + 1
    for resource in contract.resources:
        descriptor_type = _descriptor_type(vk, resource.descriptor_type)
        counts[descriptor_type] = counts.get(descriptor_type, 0) + 1
    for _uniform in contract.uniforms:
        descriptor_type = _descriptor_type(vk, "uniform_buffer")
        counts[descriptor_type] = counts.get(descriptor_type, 0) + 1
    return counts


def _specialization_info(vk: Any, constants: Mapping[int, int]) -> Any:
    entries: list[Any] = []
    values = bytearray()
    for constant_id, value in sorted(constants.items()):
        offset = len(values)
        values.extend(struct.pack("<I", value))
        entries.append(
            vk.VkSpecializationMapEntry(
                constantID=constant_id,
                offset=offset,
                size=4,
            )
        )
    value_bytes = bytes(values)
    return vk.VkSpecializationInfo(
        mapEntryCount=len(entries),
        pMapEntries=entries,
        dataSize=len(value_bytes),
        pData=vk.ffi.from_buffer(value_bytes),
    )
