"""Reusable Vulkan shader dispatch helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .logical import LogicalTensor
from .shader import (
    DispatchRecord,
    ShaderVariant,
    TensorContract,
    dispatch_dimensions,
    pack_push_constants,
    pack_uniform_blocks,
)
from .storage import StoragePlan
from .vulkan_backend import (
    VulkanBuffer,
    VulkanBufferSlice,
    VulkanComputePipeline,
    VulkanContext,
    VulkanDescriptorPool,
    VulkanDescriptorSetLayout,
    VulkanPipelineLayout,
    VulkanShaderModule,
)

type VulkanDescriptorBuffer = VulkanBuffer | VulkanBufferSlice
type LogicalTensorLookup = Mapping[str, LogicalTensor | tuple[LogicalTensor, ...]]


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
        tensor_buffers: Mapping[str, VulkanDescriptorBuffer],
        resource_buffers: Mapping[str, VulkanDescriptorBuffer] | None = None,
    ) -> None:
        symbols = self.variant.contract.validate(tensors)
        resources: Mapping[str, VulkanDescriptorBuffer] = (
            {} if resource_buffers is None else resource_buffers
        )
        uniform_buffers = _uniform_buffers(self.context, self.variant, symbols)
        try:
            descriptors: dict[int, VulkanDescriptorBuffer] = {}
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
        tensors: LogicalTensorLookup,
        tensor_buffers: Mapping[str, VulkanDescriptorBuffer],
        resource_buffers: Mapping[str, Mapping[str, VulkanDescriptorBuffer]] | None = None,
    ) -> None:
        resources: Mapping[str, Mapping[str, VulkanDescriptorBuffer]] = (
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

    def run_bound_storage(
        self,
        records: tuple[DispatchRecord, ...],
        *,
        tensors: LogicalTensorLookup,
        allocations: Mapping[str, VulkanBuffer],
        resource_buffers: Mapping[str, Mapping[str, VulkanDescriptorBuffer]] | None = None,
    ) -> None:
        tensor_buffers = storage_descriptor_buffers(
            _iter_lookup_tensors(tensors),
            allocations=allocations,
        )
        self.run(
            records,
            tensors=tensors,
            tensor_buffers=tensor_buffers,
            resource_buffers=resource_buffers,
        )


def storage_descriptor_buffers(
    tensors: Iterable[LogicalTensor],
    *,
    allocations: Mapping[str, VulkanBuffer],
) -> dict[str, VulkanBufferSlice]:
    buffers: dict[str, VulkanBufferSlice] = {}
    for tensor in tensors:
        if tensor.storage is None:
            raise ValueError(f"{tensor.name} has no bound storage")
        try:
            allocation = allocations[tensor.storage.allocation_id]
        except KeyError as exc:
            raise KeyError(
                f"Missing Vulkan allocation {tensor.storage.allocation_id} for {tensor.name}"
            ) from exc
        buffers[tensor.name] = VulkanBufferSlice(
            buffer=allocation,
            offset=tensor.storage.offset,
            nbytes=tensor.storage.nbytes,
        )
    return buffers


def allocate_storage_buffers(
    context: VulkanContext,
    plan: StoragePlan,
) -> dict[str, VulkanBuffer]:
    nbytes_by_allocation: dict[str, int] = {}
    for storage in plan.slices.values():
        end = storage.offset + storage.nbytes
        nbytes_by_allocation[storage.allocation_id] = max(
            nbytes_by_allocation.get(storage.allocation_id, 0),
            end,
        )
    return {
        allocation_id: context.create_host_buffer(nbytes=nbytes)
        for allocation_id, nbytes in sorted(nbytes_by_allocation.items())
    }


def write_bound_tensor_bytes(
    tensor: LogicalTensor,
    allocations: Mapping[str, VulkanBuffer],
    data: bytes,
) -> None:
    if tensor.storage is None:
        raise ValueError(f"{tensor.name} has no bound storage")
    if len(data) != tensor.storage.nbytes:
        raise ValueError(
            f"{tensor.name} payload has {len(data)} bytes, expected {tensor.storage.nbytes}"
        )
    try:
        allocation = allocations[tensor.storage.allocation_id]
    except KeyError as exc:
        raise KeyError(f"Missing Vulkan allocation {tensor.storage.allocation_id}") from exc
    allocation.write(data, offset=tensor.storage.offset)


def write_bound_tensor_payloads(
    tensors: LogicalTensorLookup,
    allocations: Mapping[str, VulkanBuffer],
    payloads: Mapping[str, bytes],
) -> None:
    for tensor_name, payload in payloads.items():
        try:
            tensor = _first_lookup_tensor(tensors[tensor_name])
        except KeyError as exc:
            raise KeyError(f"Missing LogicalTensor for payload {tensor_name}") from exc
        write_bound_tensor_bytes(tensor, allocations, payload)


def read_bound_tensor_bytes(
    tensor: LogicalTensor,
    allocations: Mapping[str, VulkanBuffer],
) -> bytes:
    if tensor.storage is None:
        raise ValueError(f"{tensor.name} has no bound storage")
    try:
        allocation = allocations[tensor.storage.allocation_id]
    except KeyError as exc:
        raise KeyError(f"Missing Vulkan allocation {tensor.storage.allocation_id}") from exc
    return allocation.read(offset=tensor.storage.offset, nbytes=tensor.storage.nbytes)


def _first_lookup_tensor(value: LogicalTensor | tuple[LogicalTensor, ...]) -> LogicalTensor:
    return value if isinstance(value, LogicalTensor) else value[0]


def _record_tensors(
    record: DispatchRecord,
    variant: ShaderVariant,
    tensors: LogicalTensorLookup,
) -> dict[str, LogicalTensor]:
    resolved: dict[str, LogicalTensor] = {}
    symbols: dict[str, int] = {}
    for binding in variant.contract.bindings:
        tensor_name = record.reads.get(binding.field, record.writes.get(binding.field))
        if tensor_name is None:
            continue
        contract = variant.contract.inputs.get(
            binding.field,
            variant.contract.outputs.get(binding.field),
        )
        if contract is None:
            continue
        try:
            candidate = tensors[tensor_name]
        except KeyError as exc:
            raise KeyError(f"Missing LogicalTensor {tensor_name}") from exc
        tensor, symbols = _select_tensor_view(
            tensor_name=tensor_name,
            candidate=candidate,
            contract=contract,
            symbols=symbols,
        )
        resolved[binding.field] = tensor
    return resolved


def _select_tensor_view(
    *,
    tensor_name: str,
    candidate: LogicalTensor | tuple[LogicalTensor, ...],
    contract: TensorContract,
    symbols: Mapping[str, int],
) -> tuple[LogicalTensor, dict[str, int]]:
    candidates = (candidate,) if isinstance(candidate, LogicalTensor) else candidate
    for tensor in candidates:
        matched_symbols = _match_tensor_contract(tensor, contract, symbols)
        if matched_symbols is not None:
            return tensor, matched_symbols
    raise ValueError(
        f"No LogicalTensor view for {tensor_name} matches "
        f"dtype={contract.dtype} rank={len(contract.shape)}"
    )


def _match_tensor_contract(
    tensor: LogicalTensor,
    contract: TensorContract,
    symbols: Mapping[str, int],
) -> dict[str, int] | None:
    if tensor.dtype != contract.dtype or len(tensor.shape) != len(contract.shape):
        return None
    matched = dict(symbols)
    for actual_dim, expected_dim in zip(tensor.shape, contract.shape, strict=True):
        if not isinstance(actual_dim, int):
            return None
        if isinstance(expected_dim, int):
            if actual_dim != expected_dim:
                return None
            continue
        known = matched.get(expected_dim)
        if known is not None and known != actual_dim:
            return None
        matched[expected_dim] = actual_dim
    return matched


def _iter_lookup_tensors(tensors: LogicalTensorLookup) -> tuple[LogicalTensor, ...]:
    values: list[LogicalTensor] = []
    for value in tensors.values():
        if isinstance(value, LogicalTensor):
            values.append(value)
        else:
            values.extend(value)
    return tuple(values)


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
