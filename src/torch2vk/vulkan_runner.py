"""Reusable Vulkan shader dispatch helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .logical import LogicalTensor
from .shader import (
    DispatchRecord,
    ShaderVariant,
    dispatch_dimensions,
    pack_push_constants,
    pack_uniform_blocks,
)
from .vulkan_backend import (
    VulkanBuffer,
    VulkanComputePipeline,
    VulkanContext,
    VulkanDescriptorPool,
    VulkanDescriptorSetLayout,
    VulkanPipelineLayout,
    VulkanShaderModule,
)


@dataclass(slots=True)
class VulkanShaderDispatch:
    context: VulkanContext
    variant: ShaderVariant
    module: VulkanShaderModule
    descriptor_layout: VulkanDescriptorSetLayout
    descriptor_pool: VulkanDescriptorPool
    pipeline_layout: VulkanPipelineLayout
    pipeline: VulkanComputePipeline

    @classmethod
    def load(
        cls,
        context: VulkanContext,
        variant: ShaderVariant,
        *,
        shader_dir: str | Path,
    ) -> VulkanShaderDispatch:
        spirv_path = Path(shader_dir) / f"{variant.name}.spv"
        module = context.create_shader_module(spirv_path.read_bytes())
        descriptor_layout = context.create_descriptor_set_layout(variant.contract)
        descriptor_pool = context.create_descriptor_pool(variant.contract)
        pipeline_layout = context.create_pipeline_layout(variant.contract, descriptor_layout)
        pipeline = context.create_compute_pipeline(
            shader_module=module,
            pipeline_layout=pipeline_layout,
            specialization_constants=variant.specialization_constants,
        )
        return cls(
            context=context,
            variant=variant,
            module=module,
            descriptor_layout=descriptor_layout,
            descriptor_pool=descriptor_pool,
            pipeline_layout=pipeline_layout,
            pipeline=pipeline,
        )

    def close(self) -> None:
        self.pipeline.close()
        self.pipeline_layout.close()
        self.descriptor_pool.close()
        self.descriptor_layout.close()
        self.module.close()

    def run(
        self,
        *,
        tensors: Mapping[str, LogicalTensor],
        tensor_buffers: Mapping[str, VulkanBuffer],
        resource_buffers: Mapping[str, VulkanBuffer] | None = None,
    ) -> None:
        symbols = self.variant.contract.validate(tensors)
        resources: Mapping[str, VulkanBuffer] = {} if resource_buffers is None else resource_buffers
        uniform_buffers = _uniform_buffers(self.context, self.variant, symbols)
        try:
            descriptors: dict[int, VulkanBuffer] = {}
            descriptor_types: dict[int, str] = {}
            for binding in self.variant.contract.bindings:
                tensor = tensors[binding.field]
                try:
                    descriptors[binding.binding] = tensor_buffers[tensor.name]
                except KeyError as exc:
                    raise KeyError(f"Missing Vulkan buffer for tensor {tensor.name}") from exc
                descriptor_types[binding.binding] = binding.descriptor_type
            for resource in self.variant.contract.resources:
                try:
                    descriptors[resource.binding] = resources[resource.name]
                except KeyError as exc:
                    raise KeyError(f"Missing Vulkan resource buffer {resource.name}") from exc
                descriptor_types[resource.binding] = resource.descriptor_type
            for uniform in self.variant.contract.uniforms:
                descriptors[uniform.binding] = uniform_buffers[uniform.name]
                descriptor_types[uniform.binding] = "uniform_buffer"

            descriptor_set = self.context.allocate_descriptor_set(
                descriptor_pool=self.descriptor_pool,
                descriptor_set_layout=self.descriptor_layout,
            )
            self.context.update_descriptor_set(
                descriptor_set,
                descriptors,
                descriptor_types=descriptor_types,
            )
            command_pool = self.context.create_command_pool()
            fence = self.context.create_fence()
            try:
                command_buffer = command_pool.allocate_command_buffer()
                command_buffer.begin()
                command_buffer.bind_compute_pipeline(self.pipeline)
                command_buffer.bind_descriptor_set(
                    pipeline_layout=self.pipeline_layout,
                    descriptor_set=descriptor_set,
                )
                push_constants = self.variant.contract.push_constants
                if push_constants is not None:
                    command_buffer.push_constants(
                        pipeline_layout=self.pipeline_layout,
                        data=pack_push_constants(self.variant.contract, tensors, symbols) or b"",
                    )
                x, y, z = dispatch_dimensions(self.variant.contract, symbols)
                command_buffer.dispatch(x, y, z)
                command_buffer.end()
                command_buffer.submit_and_wait(fence)
            finally:
                fence.close()
                command_pool.close()
        finally:
            for buffer in uniform_buffers.values():
                buffer.close()


@dataclass(frozen=True, slots=True)
class VulkanSequenceRunner:
    context: VulkanContext
    shader_dir: Path
    variants: Mapping[str, ShaderVariant]

    def run(
        self,
        records: tuple[DispatchRecord, ...],
        *,
        tensors: Mapping[str, LogicalTensor],
        tensor_buffers: Mapping[str, VulkanBuffer],
        resource_buffers: Mapping[str, Mapping[str, VulkanBuffer]] | None = None,
    ) -> None:
        resources: Mapping[str, Mapping[str, VulkanBuffer]] = (
            {} if resource_buffers is None else resource_buffers
        )
        for record in records:
            try:
                variant = self.variants[record.shader]
            except KeyError as exc:
                raise KeyError(f"Missing ShaderVariant for dispatch {record.shader}") from exc
            dispatch_tensors = _record_tensors(record, variant, tensors)
            dispatch = VulkanShaderDispatch.load(
                self.context,
                variant,
                shader_dir=self.shader_dir,
            )
            try:
                dispatch.run(
                    tensors=dispatch_tensors,
                    tensor_buffers=tensor_buffers,
                    resource_buffers=resources.get(record.shader),
                )
            finally:
                dispatch.close()


def _record_tensors(
    record: DispatchRecord,
    variant: ShaderVariant,
    tensors: Mapping[str, LogicalTensor],
) -> dict[str, LogicalTensor]:
    resolved: dict[str, LogicalTensor] = {}
    for binding in variant.contract.bindings:
        tensor_name = record.reads.get(binding.field, record.writes.get(binding.field))
        if tensor_name is None:
            continue
        try:
            resolved[binding.field] = tensors[tensor_name]
        except KeyError as exc:
            raise KeyError(f"Missing LogicalTensor {tensor_name}") from exc
    return resolved


def _uniform_buffers(
    context: VulkanContext,
    variant: ShaderVariant,
    symbols: Mapping[str, int],
) -> dict[str, VulkanBuffer]:
    packed = pack_uniform_blocks(variant.contract, symbols)
    buffers: dict[str, VulkanBuffer] = {}
    try:
        for name, data in packed.items():
            buffer = context.create_host_buffer(nbytes=len(data))
            buffer.write(data)
            buffers[name] = buffer
    except Exception:
        for buffer in buffers.values():
            buffer.close()
        raise
    return buffers
