"""Reusable compute pipeline and descriptor-set binding management."""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Mapping, Sequence

from vulkan import (
    VK_ACCESS_SHADER_READ_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    VK_ERROR_FRAGMENTATION,
    VK_ERROR_OUT_OF_POOL_MEMORY,
    VK_NULL_HANDLE,
    VK_OBJECT_TYPE_PIPELINE,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_SHADER_STAGE_COMPUTE_BIT,
    VkComputePipelineCreateInfo,
    VkDescriptorBufferInfo,
    VkDescriptorPoolCreateInfo,
    VkDescriptorPoolSize,
    VkDescriptorSetAllocateInfo,
    VkDescriptorSetLayoutBinding,
    VkDescriptorSetLayoutCreateInfo,
    VkPipelineLayoutCreateInfo,
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo,
    VkPipelineShaderStageCreateInfo,
    VkMemoryBarrier,
    VkShaderModuleCreateInfo,
    VkSpecializationInfo,
    VkSpecializationMapEntry,
    VkPushConstantRange,
    VkWriteDescriptorSet,
    vkAllocateDescriptorSets,
    vkCmdBindDescriptorSets,
    vkCmdBindPipeline,
    vkCmdDispatch,
    vkCmdPushConstants,
    vkCreateComputePipelines,
    vkCreateDescriptorPool,
    vkCreateDescriptorSetLayout,
    vkCreatePipelineLayout,
    vkCreateShaderModule,
    vkCmdPipelineBarrier,
    vkDestroyDescriptorPool,
    vkDestroyDescriptorSetLayout,
    vkDestroyPipeline,
    vkDestroyPipelineLayout,
    vkDestroyShaderModule,
    vkFreeDescriptorSets,
    vkUpdateDescriptorSets,
    VkCommandBufferBeginInfo,
    vkBeginCommandBuffer,
    vkEndCommandBuffer,
)
from vulkan._vulkan import ffi

from torch2vk.vulkan.allocation import BufferSlice
from torch2vk.vulkan.device import VulkanDevice
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements


VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT = 0x00000001
VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT = 0x00000002


@dataclass(frozen=True, slots=True)
class PipelineCacheKey:
    shader_spv_path: str
    descriptor_types: tuple[int, ...]
    descriptor_bindings: tuple[int, ...]
    entry_point: str
    specialization_constants: tuple[tuple[int, int], ...] | None
    push_constant_size: int
    execution_requirements: ShaderExecutionRequirements | None


def normalize_descriptor_types(
    *,
    descriptor_types: Sequence[int] | None,
    storage_buffer_count: int | None,
) -> tuple[int, ...]:
    if descriptor_types is None:
        if storage_buffer_count is None:
            raise ValueError("Either descriptor_types or storage_buffer_count must be provided")
        return (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,) * storage_buffer_count
    if storage_buffer_count is not None:
        raise ValueError("Provide either descriptor_types or storage_buffer_count, not both")
    return tuple(int(descriptor_type) for descriptor_type in descriptor_types)


def normalize_specialization_constants(
    specialization_constants: Mapping[int, int] | Sequence[int] | None,
) -> tuple[tuple[int, int], ...] | None:
    if specialization_constants is None:
        return None
    if isinstance(specialization_constants, Mapping):
        return tuple(
            sorted((int(constant_id), int(value)) for constant_id, value in specialization_constants.items())
        )
    return tuple((index, int(value)) for index, value in enumerate(specialization_constants))


class ComputePipeline:
    """One reusable compute pipeline with explicit descriptor bindings."""

    def __init__(
        self,
        device: VulkanDevice,
        *,
        shader_spv_path: str | Path,
        descriptor_types: Sequence[int] | None = None,
        descriptor_bindings: Sequence[int] | None = None,
        storage_buffer_count: int | None = None,
        entry_point: str = "main",
        specialization_constants: Mapping[int, int] | Sequence[int] | None = None,
        push_constant_size: int = 0,
        execution_requirements: ShaderExecutionRequirements | None = None,
    ) -> None:
        self.device = device
        self.shader_spv_path = Path(shader_spv_path)
        self.push_constant_size = push_constant_size
        self.execution_requirements = execution_requirements
        self.descriptor_types = normalize_descriptor_types(
            descriptor_types=descriptor_types,
            storage_buffer_count=storage_buffer_count,
        )
        if descriptor_bindings is None:
            self.descriptor_bindings = tuple(range(len(self.descriptor_types)))
        else:
            self.descriptor_bindings = tuple(int(binding) for binding in descriptor_bindings)
        if len(self.descriptor_bindings) != len(self.descriptor_types):
            raise ValueError(
                f"descriptor_bindings length {len(self.descriptor_bindings)} must match "
                f"descriptor_types length {len(self.descriptor_types)}"
            )
        self.device.require_shader_execution_requirements(self.execution_requirements)

        code = self.shader_spv_path.read_bytes()
        self.shader_spv_sha256 = hashlib.sha256(code).hexdigest()
        self.pipeline_identity_sha256 = _build_pipeline_identity_sha256(
            shader_spv_path=self.shader_spv_path,
            shader_spv_sha256=self.shader_spv_sha256,
            descriptor_types=self.descriptor_types,
            descriptor_bindings=self.descriptor_bindings,
            entry_point=entry_point,
            specialization_constants=normalize_specialization_constants(specialization_constants),
            push_constant_size=push_constant_size,
            execution_requirements=execution_requirements,
        )
        self.debug_name = _build_pipeline_debug_name(
            shader_stem=self.shader_spv_path.stem,
            pipeline_identity_sha256=self.pipeline_identity_sha256,
        )
        self.shader_module = vkCreateShaderModule(
            self.device.device,
            VkShaderModuleCreateInfo(codeSize=len(code), pCode=code),
            None,
        )

        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=binding_index,
                descriptorType=descriptor_type,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for binding_index, descriptor_type in zip(self.descriptor_bindings, self.descriptor_types, strict=True)
        ]
        self.descriptor_set_layout = vkCreateDescriptorSetLayout(
            self.device.device,
            VkDescriptorSetLayoutCreateInfo(
                bindingCount=len(bindings),
                pBindings=bindings,
            ),
            None,
        )

        push_ranges: list[object] = []
        if push_constant_size > 0:
            push_ranges.append(
                VkPushConstantRange(
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                    offset=0,
                    size=push_constant_size,
                )
            )
        self.pipeline_layout = vkCreatePipelineLayout(
            self.device.device,
            VkPipelineLayoutCreateInfo(
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
                pushConstantRangeCount=len(push_ranges),
                pPushConstantRanges=push_ranges or None,
            ),
            None,
        )

        specialization_info = None
        if specialization_constants is not None:
            items = normalize_specialization_constants(specialization_constants)
            assert items is not None
            spec_data = struct.pack(f"{len(items)}i", *(value for _, value in items))
            spec_entries = [
                VkSpecializationMapEntry(constantID=constant_id, offset=index * 4, size=4)
                for index, (constant_id, _) in enumerate(items)
            ]
            self._specialization_data = spec_data
            self._specialization_entries = spec_entries
            specialization_info = VkSpecializationInfo(
                mapEntryCount=len(spec_entries),
                pMapEntries=spec_entries,
                dataSize=len(spec_data),
                pData=ffi.from_buffer(spec_data),
            )
        else:
            self._specialization_data = None
            self._specialization_entries = None

        stage_flags = 0
        stage_pnext = None
        required_subgroup_info = None
        subgroup_requirements = None if self.execution_requirements is None else self.execution_requirements.subgroup
        if subgroup_requirements is not None:
            stage_flags |= VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT
            required_subgroup_info = VkPipelineShaderStageRequiredSubgroupSizeCreateInfo(
                requiredSubgroupSize=subgroup_requirements.required_size,
            )
            stage_pnext = ffi.addressof(required_subgroup_info)
        if subgroup_requirements is not None and subgroup_requirements.require_full_subgroups:
            stage_flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT

        stage = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_module,
            pName=entry_point,
            pSpecializationInfo=specialization_info,
            flags=stage_flags,
            pNext=stage_pnext,
        )
        self.pipeline = vkCreateComputePipelines(
            self.device.device,
            VK_NULL_HANDLE,
            1,
            [VkComputePipelineCreateInfo(stage=stage, layout=self.pipeline_layout)],
            None,
        )[0]
        self.device.set_debug_name(
            object_type=VK_OBJECT_TYPE_PIPELINE,
            handle=self.pipeline,
            name=self.debug_name,
        )
        self._descriptor_pool_counts = tuple(sorted(Counter(self.descriptor_types).items()))
        self._descriptor_pool_growth = 64
        self._descriptor_pools: list[object] = [self._create_descriptor_pool(self._descriptor_pool_growth)]

    def close(self) -> None:
        for descriptor_pool in self._descriptor_pools:
            if descriptor_pool != VK_NULL_HANDLE:
                vkDestroyDescriptorPool(self.device.device, descriptor_pool, None)
        self._descriptor_pools = []
        if self.pipeline != VK_NULL_HANDLE:
            vkDestroyPipeline(self.device.device, self.pipeline, None)
            self.pipeline = VK_NULL_HANDLE
        if self.pipeline_layout != VK_NULL_HANDLE:
            vkDestroyPipelineLayout(self.device.device, self.pipeline_layout, None)
            self.pipeline_layout = VK_NULL_HANDLE
        if self.descriptor_set_layout != VK_NULL_HANDLE:
            vkDestroyDescriptorSetLayout(self.device.device, self.descriptor_set_layout, None)
            self.descriptor_set_layout = VK_NULL_HANDLE
        if self.shader_module != VK_NULL_HANDLE:
            vkDestroyShaderModule(self.device.device, self.shader_module, None)
            self.shader_module = VK_NULL_HANDLE

    def __enter__(self) -> "ComputePipeline":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def dispatch(
        self,
        *,
        buffers: Sequence["DescriptorBufferBinding"],
        group_count_x: int,
        group_count_y: int = 1,
        group_count_z: int = 1,
        push_constants: bytes | None = None,
    ) -> None:
        self.device.require_open()
        if len(buffers) != len(self.descriptor_types):
            raise ValueError(f"Expected {len(self.descriptor_types)} bound buffers, got {len(buffers)}")
        self._validate_push_constants(push_constants)

        with self.bind_buffers(buffers) as binding:
            command_buffer = self.device.allocate_command_buffer()
            vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo())
            self.record_dispatch(
                command_buffer=command_buffer,
                binding=binding,
                group_count_x=group_count_x,
                group_count_y=group_count_y,
                group_count_z=group_count_z,
                push_constants=push_constants,
            )
            self.record_eager_completion_barrier(command_buffer)
            vkEndCommandBuffer(command_buffer)
            self.device.submit_one_shot_and_wait(command_buffer)

    def bind_buffers(self, buffers: Sequence["DescriptorBufferBinding"]) -> "BoundComputeBinding":
        self.device.require_open()
        if len(buffers) != len(self.descriptor_types):
            raise ValueError(f"Expected {len(self.descriptor_types)} bound buffers, got {len(buffers)}")
        descriptor_set, descriptor_pool = self._allocate_descriptor_set()
        buffer_infos = [
            VkDescriptorBufferInfo(buffer=buffer.buffer.handle, offset=buffer.offset, range=buffer.range)
            for buffer in buffers
        ]
        writes = [
            VkWriteDescriptorSet(
                dstSet=descriptor_set,
                dstBinding=binding_index,
                descriptorCount=1,
                descriptorType=descriptor_type,
                pBufferInfo=[buffer_info],
            )
            for binding_index, descriptor_type, buffer_info in zip(
                self.descriptor_bindings,
                self.descriptor_types,
                buffer_infos,
                strict=True,
            )
        ]
        vkUpdateDescriptorSets(self.device.device, len(writes), writes, 0, None)
        return BoundComputeBinding(
            pipeline=self,
            descriptor_pool=descriptor_pool,
            descriptor_set=descriptor_set,
        )

    def record_dispatch(
        self,
        *,
        command_buffer: object,
        binding: "BoundComputeBinding",
        group_count_x: int,
        group_count_y: int = 1,
        group_count_z: int = 1,
        push_constants: bytes | None = None,
    ) -> None:
        self.device.require_open()
        self._validate_push_constants(push_constants)
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
            0,
            1,
            [binding.descriptor_set],
            0,
            None,
        )
        if push_constants is not None:
            vkCmdPushConstants(
                command_buffer,
                self.pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_constants),
                ffi.from_buffer(push_constants),
            )
        vkCmdDispatch(command_buffer, group_count_x, group_count_y, group_count_z)

    def _validate_push_constants(self, push_constants: bytes | None) -> None:
        if self.push_constant_size == 0 and push_constants is not None:
            raise ValueError("Pipeline does not declare push constants")
        if self.push_constant_size > 0:
            if push_constants is None:
                raise ValueError("Pipeline requires push constants")
            if len(push_constants) != self.push_constant_size:
                raise ValueError(
                    f"Expected {self.push_constant_size} push-constant bytes, got {len(push_constants)}"
                )

    def record_eager_completion_barrier(self, command_buffer: object) -> None:
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            [
                VkMemoryBarrier(
                    srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                    dstAccessMask=VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                )
            ],
            0,
            None,
            0,
            None,
        )

    def _allocate_descriptor_set(self) -> tuple[object, object]:
        for descriptor_pool in reversed(self._descriptor_pools):
            descriptor_set = self._try_allocate_descriptor_set(descriptor_pool)
            if descriptor_set is not None:
                return descriptor_set, descriptor_pool

        next_capacity = self._descriptor_pool_growth * 2
        descriptor_pool = self._create_descriptor_pool(next_capacity)
        self._descriptor_pools.append(descriptor_pool)
        self._descriptor_pool_growth = next_capacity
        descriptor_set = self._try_allocate_descriptor_set(descriptor_pool)
        if descriptor_set is None:
            raise RuntimeError("Newly created descriptor pool could not allocate a descriptor set")
        return descriptor_set, descriptor_pool

    def _try_allocate_descriptor_set(self, descriptor_pool: object) -> object | None:
        try:
            return vkAllocateDescriptorSets(
                self.device.device,
                VkDescriptorSetAllocateInfo(
                    descriptorPool=descriptor_pool,
                    descriptorSetCount=1,
                    pSetLayouts=[self.descriptor_set_layout],
                ),
            )[0]
        except KeyError as exc:
            error_code = exc.args[0] if exc.args else None
            if error_code in {
                -VK_ERROR_OUT_OF_POOL_MEMORY,
                VK_ERROR_OUT_OF_POOL_MEMORY,
                -VK_ERROR_FRAGMENTATION,
                VK_ERROR_FRAGMENTATION,
            }:
                return None
            raise

    def _create_descriptor_pool(self, max_sets: int) -> object:
        pool_sizes = [
            VkDescriptorPoolSize(type=descriptor_type, descriptorCount=count * max_sets)
            for descriptor_type, count in self._descriptor_pool_counts
        ]
        return vkCreateDescriptorPool(
            self.device.device,
            VkDescriptorPoolCreateInfo(
                flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                maxSets=max_sets,
                poolSizeCount=len(pool_sizes),
                pPoolSizes=pool_sizes,
            ),
            None,
        )


@dataclass(slots=True)
class BoundComputeBinding:
    pipeline: ComputePipeline
    descriptor_pool: object
    descriptor_set: object

    @property
    def closed(self) -> bool:
        return self.descriptor_set == VK_NULL_HANDLE

    def close(self) -> None:
        if self.closed:
            return
        if self.pipeline.device.closed:
            self.descriptor_set = VK_NULL_HANDLE
            return
        vkFreeDescriptorSets(
            self.pipeline.device.device,
            self.descriptor_pool,
            1,
            [self.descriptor_set],
        )
        self.descriptor_set = VK_NULL_HANDLE

    def __enter__(self) -> "BoundComputeBinding":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class DescriptorBufferBinding:
    slice: BufferSlice

    def __post_init__(self) -> None:
        if self.slice.nbytes <= 0:
            raise ValueError(f"DescriptorBufferBinding range must be positive, got {self.slice.nbytes}")

    @classmethod
    def from_slice(cls, slice: BufferSlice, *, descriptor_nbytes: int | None = None) -> "DescriptorBufferBinding":
        nbytes = slice.nbytes if descriptor_nbytes is None else int(descriptor_nbytes)
        return cls(
            slice=BufferSlice(
                allocation=slice.allocation,
                offset=slice.offset,
                nbytes=nbytes,
            ),
        )

    def matches_slice(self, slice: BufferSlice, *, descriptor_nbytes: int | None = None) -> bool:
        nbytes = slice.nbytes if descriptor_nbytes is None else int(descriptor_nbytes)
        return (
            self.slice.allocation is slice.allocation
            and self.slice.offset == slice.offset
            and self.slice.nbytes == nbytes
        )

    @property
    def buffer(self):
        return self.slice.allocation.buffer

    @property
    def offset(self) -> int:
        return self.slice.offset

    @property
    def range(self) -> int:
        return self.slice.nbytes


def _build_pipeline_identity_sha256(
    *,
    shader_spv_path: Path,
    shader_spv_sha256: str,
    descriptor_types: tuple[int, ...],
    descriptor_bindings: tuple[int, ...],
    entry_point: str,
    specialization_constants: tuple[tuple[int, int], ...] | None,
    push_constant_size: int,
    execution_requirements: ShaderExecutionRequirements | None,
) -> str:
    subgroup = None if execution_requirements is None else execution_requirements.subgroup
    cooperative_matrix = None if execution_requirements is None else execution_requirements.cooperative_matrix
    payload = {
        "shader_spv_path": str(shader_spv_path),
        "shader_spv_sha256": shader_spv_sha256,
        "descriptor_types": list(descriptor_types),
        "descriptor_bindings": list(descriptor_bindings),
        "entry_point": entry_point,
        "specialization_constants": None
        if specialization_constants is None
        else [[constant_id, value] for constant_id, value in specialization_constants],
        "push_constant_size": push_constant_size,
        "subgroup": None
        if subgroup is None
        else {
            "required_size": subgroup.required_size,
            "require_full_subgroups": subgroup.require_full_subgroups,
        },
        "cooperative_matrix": None
        if cooperative_matrix is None
        else {
            "scope": cooperative_matrix.scope,
            "m_size": cooperative_matrix.m_size,
            "n_size": cooperative_matrix.n_size,
            "k_size": cooperative_matrix.k_size,
            "a_type": cooperative_matrix.a_type,
            "b_type": cooperative_matrix.b_type,
            "c_type": cooperative_matrix.c_type,
            "result_type": cooperative_matrix.result_type,
            "saturating_accumulation": cooperative_matrix.saturating_accumulation,
        },
        "require_integer_dot_product": False
        if execution_requirements is None
        else execution_requirements.require_integer_dot_product,
        "require_shader_int64": False
        if execution_requirements is None
        else execution_requirements.require_shader_int64,
        "require_buffer_device_address": False
        if execution_requirements is None
        else execution_requirements.require_buffer_device_address,
        "require_storage_buffer_16bit_access": False
        if execution_requirements is None
        else execution_requirements.require_storage_buffer_16bit_access,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_pipeline_debug_name(*, shader_stem: str, pipeline_identity_sha256: str) -> str:
    suffix = pipeline_identity_sha256[:16]
    max_total_length = 63
    prefix = "agp."
    separator = "."
    max_stem_length = max_total_length - len(prefix) - len(separator) - len(suffix)
    if max_stem_length <= 0:
        raise ValueError("pipeline debug-name budget is invalid")
    normalized_stem = shader_stem[:max_stem_length]
    return f"{prefix}{normalized_stem}.{suffix}"
