"""Replay plan instantiation and command-buffer recording."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

from vulkan import (
    VK_ACCESS_HOST_READ_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_ACCESS_TRANSFER_READ_BIT,
    VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_HOST_BIT,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VkBufferCopy,
    VkCommandBufferBeginInfo,
    VkFenceCreateInfo,
    VkMemoryBarrier,
    vkBeginCommandBuffer,
    vkCmdCopyBuffer,
    vkCmdPipelineBarrier,
    vkCmdResetQueryPool,
    vkCmdWriteTimestamp,
    vkCreateFence,
    vkEndCommandBuffer,
)

from torch2vk.runtime.host_array import prepare_host_array
from torch2vk.runtime.logical import LogicalTensor, MemoryClass
from torch2vk.runtime.replay import (
    ReplayDescriptorBinding,
    ReplayDispatchEntry,
    ReplayDispatchTemplate,
    ReplayPlan,
    ReplayPlanTemplate,
    ReplayProfileState,
    ReplayReadbackSlot,
)
from torch2vk.runtime.replay_descriptor import (
    canonical_replay_descriptor_tensor,
    has_live_buffer,
    replay_descriptor_rebindable,
)
from torch2vk.runtime.replay_template import (
    build_replay_template_from_records,
    collect_replay_tensors,
    frame_dispatch_records,
)
from torch2vk.runtime.shader import (
    ParamsBufferFieldSpec,
    ParamsBufferSpec,
    PushConstantFieldSpec,
    PushConstantInput,
    PushConstantSpec,
    PushConstantType,
    PushConstantValue,
    ShaderContract,
    ShaderVariant,
    referenced_symbols,
)
from torch2vk.vulkan.allocation import BufferAllocation, BufferSlice
from torch2vk.vulkan.compute_pipeline import BoundComputeBinding, DescriptorBufferBinding
from torch2vk.vulkan.types import tensor_nbytes

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


_PUSH_CONSTANT_DECL_RE = re.compile(
    r"layout\s*\(\s*push_constant\s*\)\s*uniform\s+\w+\s*\{.*?\}\s*pc\s*;",
    re.DOTALL,
)


def build_replay_plan(
    rt: RuntimeSession,
    *,
    name: str,
    frame: str,
    readback_tensors: Mapping[str, LogicalTensor] | None = None,
) -> ReplayPlan:
    """Build a ReplayPlan from previously recorded dispatch information."""
    rt._require_open()
    records = frame_dispatch_records(rt, frame)
    if not records:
        raise ValueError(f"Cannot build replay plan for frame {frame!r} with zero dispatches")
    logical_tensors = collect_replay_tensors(rt, records)
    template = build_replay_template_from_records(
        rt,
        name=name,
        frame_dispatch_records=records,
        logical_tensors=logical_tensors,
    )
    return instantiate_replay_template(
        rt,
        template=template,
        logical_tensors=logical_tensors,
        readback_tensors=readback_tensors,
    )


def instantiate_replay_template(
    rt: RuntimeSession,
    *,
    template: ReplayPlanTemplate,
    logical_tensors: Mapping[str, LogicalTensor],
    readback_tensors: Mapping[str, LogicalTensor] | None = None,
) -> ReplayPlan:
    num_dispatches = len(template.entries)
    dynamic_symbol_names = template.dynamic_symbol_names
    dynamic_push_constant_names = template.dynamic_push_constant_names

    use_indirect_dispatch = len(dynamic_symbol_names) > 0
    indirect_buffer: BufferAllocation | None = None
    if use_indirect_dispatch:
        indirect_buffer = rt.device.allocate_host_visible_allocation(
            num_dispatches * 12,
            usage_flags=(
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            ),
        )
        indirect_buffer.buffer.map_persistent()

    dispatch_entries: list[ReplayDispatchEntry] = []
    params_entries: list[ReplayDispatchEntry] = []
    bindings: list[BoundComputeBinding] = []
    workspace_allocations: list[BufferAllocation] = []
    params_allocations: list[BufferAllocation] = []
    workspace_pool = _ReplayWorkspacePool(logical_tensors)
    tensor_last_use = _replay_tensor_last_use(
        template.entries,
        logical_tensors=logical_tensors,
    )

    for i, template_entry in enumerate(template.entries):
        source_variant = rt._model_shader(template_entry.shader)
        dynamic_symbols = template_entry.dynamic_symbol_names
        dynamic_push_constants = template_entry.dynamic_push_constant_names
        variant = _replay_variant_for_dynamic_symbols(
            source_variant,
            dynamic_symbols,
            dynamic_push_constants,
        )
        contract = variant.contract
        pipeline = rt._pipeline_for_variant(variant)
        record_symbols = dict(template_entry.symbols)
        logical_by_field = dict(template_entry.logical_reads)
        logical_by_field.update(template_entry.logical_writes)
        tensors = {
            field.name: logical_tensors[logical_by_field[field.name]] for field in contract.fields
        }

        buffer_views: list[DescriptorBufferBinding] = []
        descriptor_bindings: list[ReplayDescriptorBinding] = []
        for field in contract.fields:
            tensor = tensors[field.name]
            descriptor_tensor = canonical_replay_descriptor_tensor(
                tensor=tensor,
                logical_tensors=logical_tensors,
            )
            descriptor_rebindable = replay_descriptor_rebindable(descriptor_tensor)
            if not has_live_buffer(descriptor_tensor):
                if descriptor_tensor.alias_source is not None:
                    rt._materialize_read(descriptor_tensor)
                elif descriptor_tensor.memory is MemoryClass.MODEL_WEIGHT:
                    rt._materialize_weight(descriptor_tensor)
                elif descriptor_tensor.memory is MemoryClass.SESSION_TENSOR:
                    raise RuntimeError(
                        f"{descriptor_tensor.name} requires initialize_session_tensors(...) "
                        "before building or instantiating replay"
                    )
                else:
                    alloc = workspace_pool.acquire(rt, descriptor_tensor)
                    with descriptor_tensor.runtime_write_scope():
                        descriptor_tensor.buffer = BufferSlice(
                            allocation=alloc,
                            offset=alloc.offset,
                            nbytes=tensor_nbytes(descriptor_tensor.spec),
                        )
                        descriptor_tensor.descriptor_nbytes = descriptor_tensor.buffer.nbytes
                    if not any(existing is alloc for existing in workspace_allocations):
                        workspace_allocations.append(alloc)

            descriptor_buffer = descriptor_tensor.buffer
            if descriptor_buffer is None:
                raise RuntimeError(f"{descriptor_tensor.name} is not materialized")
            descriptor_binding = DescriptorBufferBinding.from_slice(
                descriptor_buffer,
                descriptor_nbytes=descriptor_tensor.descriptor_nbytes,
            )
            buffer_views.append(descriptor_binding)
            descriptor_bindings.append(
                ReplayDescriptorBinding(
                    field=field,
                    tensor=descriptor_tensor,
                    tensor_name=descriptor_tensor.name,
                    buffer=descriptor_binding,
                    rebindable=descriptor_rebindable,
                    validate_shape=descriptor_tensor is tensor,
                )
            )

        params_alloc: BufferAllocation | None = None
        if contract.params_buffer is not None:
            params_alloc = rt._materialize_params_buffer(
                contract.params_buffer,
                tensors=tensors,
                symbols=record_symbols,
                push_constant_inputs=dict(template_entry.push_constant_values),
            )
            params_alloc.buffer.map_persistent()
            params_slice = BufferSlice(
                allocation=params_alloc,
                offset=params_alloc.offset,
                nbytes=contract.params_buffer.size,
            )
            buffer_views.append(DescriptorBufferBinding(slice=params_slice))
            params_allocations.append(params_alloc)

        binding = pipeline.bind_buffers(buffer_views)
        bindings.append(binding)

        push_constants, _ = rt._pack_push_constants(
            contract.push_constants,
            tensors=tensors,
            symbols=record_symbols,
            push_constant_inputs=dict(template_entry.push_constant_values),
        )

        entry = ReplayDispatchEntry(
            pipeline=pipeline,
            binding=binding,
            descriptors=tuple(descriptor_bindings),
            push_constants=push_constants,
            dispatch_size=template_entry.dispatch_size,
            dispatch_formula=contract.dispatch,
            symbols=record_symbols,
            push_constant_values=dict(template_entry.push_constant_values),
            dynamic_symbol_names=dynamic_symbols,
            dynamic_push_constant_names=dynamic_push_constants,
            indirect_offset=i * 12 if use_indirect_dispatch else None,
            params_buffer=params_alloc,
            params_layout=contract.params_buffer,
            source_dispatch_index=template_entry.source_dispatch_index,
            source_frame=template_entry.source_frame,
            source_shader=template_entry.shader,
            source_logical_reads=template_entry.logical_reads,
            source_logical_writes=template_entry.logical_writes,
        )
        dispatch_entries.append(entry)
        if params_alloc is not None:
            params_entries.append(entry)
        workspace_pool.recycle_after_dispatch(i, tensor_last_use)

    readback_slots: dict[str, ReplayReadbackSlot] = {}
    readback_copy_sources: list[tuple[str, BufferSlice, int]] = []
    if readback_tensors:
        for rname, rtensor in readback_tensors.items():
            if rtensor.buffer is None:
                raise RuntimeError(f"Readback tensor {rtensor.name} not materialized")
            nbytes = rtensor.buffer.nbytes
            slot_alloc = rt.device.allocate_host_visible_allocation(
                nbytes,
                usage_flags=VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            )
            readback_slots[rname] = ReplayReadbackSlot(
                name=rname,
                allocation=slot_alloc,
                nbytes=nbytes,
            )
            readback_copy_sources.append((rname, rtensor.buffer, nbytes))

    command_buffer = rt.device.allocate_command_buffer()
    profile_state = (
        ReplayProfileState.create(rt.device, num_dispatches=len(dispatch_entries))
        if rt.profiler.enabled
        else None
    )
    vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo())
    if profile_state is not None:
        assert profile_state.query_pool is not None
        vkCmdResetQueryPool(command_buffer, profile_state.query_pool, 0, profile_state.query_count)
        vkCmdWriteTimestamp(
            command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            profile_state.query_pool,
            0,
        )

    for i, entry in enumerate(dispatch_entries):
        if profile_state is not None:
            assert profile_state.query_pool is not None
            vkCmdWriteTimestamp(
                command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                profile_state.query_pool,
                1 + i * 2,
            )
        if indirect_buffer is None:
            entry.pipeline.record_dispatch(
                command_buffer=command_buffer,
                binding=entry.binding,
                group_count_x=entry.dispatch_size[0],
                group_count_y=entry.dispatch_size[1],
                group_count_z=entry.dispatch_size[2],
                push_constants=entry.push_constants,
            )
        else:
            if entry.indirect_offset is None:
                raise RuntimeError(
                    f"Replay entry for {entry.pipeline.debug_name} has no indirect offset"
                )
            entry.pipeline.record_indirect_dispatch(
                command_buffer=command_buffer,
                binding=entry.binding,
                indirect_buffer=indirect_buffer.buffer.handle,
                indirect_offset=indirect_buffer.offset + entry.indirect_offset,
                push_constants=entry.push_constants,
            )
        if profile_state is not None:
            assert profile_state.query_pool is not None
            vkCmdWriteTimestamp(
                command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                profile_state.query_pool,
                2 + i * 2,
            )
        entry.pipeline.record_eager_completion_barrier(command_buffer)

    if readback_copy_sources:
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            1,
            [
                VkMemoryBarrier(
                    srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                    dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
                )
            ],
            0,
            None,
            0,
            None,
        )
        for rname, source_slice, nbytes in readback_copy_sources:
            slot = readback_slots[rname]
            vkCmdCopyBuffer(
                command_buffer,
                source_slice.allocation.buffer.handle,
                slot.allocation.buffer.handle,
                1,
                [
                    VkBufferCopy(
                        srcOffset=source_slice.offset,
                        dstOffset=slot.allocation.offset,
                        size=nbytes,
                    )
                ],
            )
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1,
            [
                VkMemoryBarrier(
                    srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=VK_ACCESS_HOST_READ_BIT,
                )
            ],
            0,
            None,
            0,
            None,
        )

    if profile_state is not None:
        assert profile_state.query_pool is not None
        vkCmdWriteTimestamp(
            command_buffer,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            profile_state.query_pool,
            profile_state.query_count - 1,
        )
    vkEndCommandBuffer(command_buffer)
    fence = vkCreateFence(rt.device.device, VkFenceCreateInfo(), None)

    return ReplayPlan(
        device=rt.device,
        name=template.name,
        command_buffer=command_buffer,
        fence=fence,
        indirect_buffer=indirect_buffer,
        num_dispatches=num_dispatches,
        dispatch_entries=tuple(dispatch_entries),
        params_entries=tuple(params_entries),
        dynamic_symbol_names=dynamic_symbol_names,
        dynamic_push_constant_names=dynamic_push_constant_names,
        template=template,
        readback_slots=readback_slots,
        workspace_allocations=workspace_allocations + params_allocations,
        bindings=bindings,
        profile_state=profile_state,
        profile_recorder=rt.profiler,
    )


def _replay_variant_for_dynamic_symbols(
    variant: ShaderVariant,
    dynamic_symbol_names: tuple[str, ...],
    dynamic_push_constant_names: tuple[str, ...],
) -> ShaderVariant:
    push_constants = variant.contract.push_constants
    if push_constants is None:
        return variant

    dynamic_symbols = set(dynamic_symbol_names)
    dynamic_push_constants = set(dynamic_push_constant_names)
    dynamic_fields = tuple(
        field
        for field in push_constants.fields
        if dynamic_symbols.intersection(_push_constant_value_symbols(field.value))
        or (
            isinstance(field.value, PushConstantInput)
            and field.value.name in dynamic_push_constants
        )
    )
    if not dynamic_fields:
        return variant
    if variant.contract.params_buffer is not None:
        raise RuntimeError(
            f"Replay shader {variant.name!r} cannot move dynamic push constants "
            "into params because it already has a params buffer"
        )
    if not variant.source:
        raise RuntimeError(
            f"Replay shader {variant.name!r} needs inline GLSL source for dynamic push constants"
        )

    dynamic_field_names = {field.name for field in dynamic_fields}
    static_push_fields: list[PushConstantFieldSpec] = []
    dynamic_param_fields: list[ParamsBufferFieldSpec] = []
    static_offset = 0
    params_offset = 0
    for field in push_constants.fields:
        if field.name in dynamic_field_names:
            _validate_replay_param_value(variant.name, field)
            params_offset = _align_push_constant_offset(params_offset, field.dtype)
            dynamic_param_fields.append(
                ParamsBufferFieldSpec(
                    name=field.name,
                    dtype=field.dtype,
                    offset=params_offset,
                    value=field.value,
                )
            )
            params_offset += field.size
            continue
        static_offset = _align_push_constant_offset(static_offset, field.dtype)
        static_push_fields.append(
            PushConstantFieldSpec(
                name=field.name,
                dtype=field.dtype,
                offset=static_offset,
                value=field.value,
                dynamic=field.dynamic,
            )
        )
        static_offset += field.size

    replay_name = f"{variant.name}__replay_dynamic"
    replay_source = _rewrite_replay_push_constants(
        shader_name=variant.name,
        source=variant.source,
        static_push_fields=tuple(static_push_fields),
        dynamic_param_fields=tuple(dynamic_param_fields),
        params_binding_index=len(variant.contract.fields),
    )
    replay_contract = ShaderContract(
        class_name=variant.contract.class_name,
        shader_name=replay_name,
        fields=variant.contract.fields,
        dispatch=variant.contract.dispatch,
        push_constants=(
            PushConstantSpec(
                size=static_offset,
                fields=tuple(static_push_fields),
            )
            if static_push_fields
            else None
        ),
        params_buffer=ParamsBufferSpec(
            size=params_offset,
            fields=tuple(dynamic_param_fields),
            binding_index=len(variant.contract.fields),
        ),
    )
    return ShaderVariant(
        name=replay_name,
        family=variant.family,
        contract=replay_contract,
        source=replay_source,
        precompiled_spv_path=None,
        specialization_constants=variant.specialization_constants,
        include_dirs=variant.include_dirs,
        compile_defines=variant.compile_defines,
        execution_requirements=variant.execution_requirements,
    )


def _push_constant_value_symbols(value: PushConstantValue) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, int | float):
        return ()
    if isinstance(value, PushConstantInput):
        return ()
    if callable(value):
        return ()
    return referenced_symbols(value)


def _validate_replay_param_value(
    shader_name: str,
    field: PushConstantFieldSpec,
) -> None:
    if callable(field.value):
        raise RuntimeError(
            f"Replay shader {shader_name!r} cannot move callable push constant "
            f"{field.name!r} into params"
        )
    if field.dtype is PushConstantType.UINT64:
        raise RuntimeError(
            f"Replay shader {shader_name!r} cannot move uint64 push constant "
            f"{field.name!r} into params"
        )


def _align_push_constant_offset(offset: int, dtype: PushConstantType) -> int:
    alignment = 8 if dtype is PushConstantType.UINT64 else 4
    return ((offset + alignment - 1) // alignment) * alignment


def _rewrite_replay_push_constants(
    *,
    shader_name: str,
    source: str,
    static_push_fields: tuple[PushConstantFieldSpec, ...],
    dynamic_param_fields: tuple[ParamsBufferFieldSpec, ...],
    params_binding_index: int,
) -> str:
    replacement_parts: list[str] = []
    if static_push_fields:
        replacement_parts.append(_push_constant_decl_source(static_push_fields))
    replacement_parts.append(
        _replay_params_decl_source(dynamic_param_fields, binding_index=params_binding_index)
    )
    rewritten, count = _PUSH_CONSTANT_DECL_RE.subn("\n".join(replacement_parts), source, count=1)
    if count != 1:
        raise RuntimeError(
            f"Replay shader {shader_name!r} could not find a single push constant block named pc"
        )
    for field in dynamic_param_fields:
        rewritten = re.sub(
            rf"\bpc\.{re.escape(field.name)}\b",
            f"_replay_params.{field.name}",
            rewritten,
        )
    return rewritten


def _push_constant_decl_source(fields: tuple[PushConstantFieldSpec, ...]) -> str:
    lines = ["layout(push_constant) uniform PushConstants {"]
    for field in fields:
        lines.append(f"    {_glsl_scalar_type(field.dtype)} {field.name};")
    lines.append("} pc;")
    return "\n".join(lines)


def _replay_params_decl_source(
    fields: tuple[ParamsBufferFieldSpec, ...],
    *,
    binding_index: int,
) -> str:
    lines = [
        f"layout(set = 0, binding = {binding_index}) buffer restrict readonly ReplayParamsBuffer {{"
    ]
    for field in fields:
        lines.append(f"    {_glsl_scalar_type(field.dtype)} {field.name};")
    lines.append("} _replay_params;")
    return "\n".join(lines)


def _glsl_scalar_type(dtype: PushConstantType) -> str:
    if dtype is PushConstantType.UINT32:
        return "uint"
    if dtype is PushConstantType.INT32:
        return "int"
    if dtype is PushConstantType.FLOAT32:
        return "float"
    if dtype is PushConstantType.UINT64:
        return "uint64_t"
    raise ValueError(f"Unsupported push constant dtype {dtype!r}")


def _allocate_replay_descriptor_tensor(
    rt: RuntimeSession,
    descriptor_tensor: LogicalTensor,
) -> BufferAllocation:
    nbytes = tensor_nbytes(descriptor_tensor.spec)
    if nbytes == 0:
        raise RuntimeError(
            f"Tensor {descriptor_tensor.name} has zero size, cannot build replay plan"
        )
    if descriptor_tensor.memory is MemoryClass.HOST_INPUT:
        alloc = rt.device.allocate_host_visible_allocation(nbytes)
        alloc.buffer.map_persistent()
        value = rt._inputs.get(descriptor_tensor)
        if value is not None:
            array = prepare_host_array(
                descriptor_tensor,
                value,
                context="replay input",
            )
            if array.nbytes != nbytes:
                raise ValueError(
                    f"{descriptor_tensor.name} replay input has {array.nbytes} bytes, "
                    f"expected {nbytes}"
                )
            alloc.buffer.write_bytes_at(alloc.offset, memoryview(array).cast("B"))
            rt.device.memory_manager.host_upload_ring.flush(allocation=alloc, size=nbytes)
        return alloc
    if descriptor_tensor.memory is MemoryClass.SESSION_TENSOR:
        raise RuntimeError(
            f"{descriptor_tensor.name} requires initialize_session_tensors(...) before replay"
        )

    if descriptor_tensor.memory in {
        MemoryClass.FRAME_WORKSPACE,
        MemoryClass.OP_SCRATCH,
        MemoryClass.REQUEST_STATE,
        MemoryClass.HOST_OUTPUT,
    }:
        return rt.device.memory_manager.allocate_device_local_buffer(
            nbytes,
            usage_flags=(
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            ),
        )

    raise RuntimeError(
        f"{descriptor_tensor.name} is not materialized and cannot be allocated as replay workspace"
    )


class _ReplayWorkspacePool:
    def __init__(self, logical_tensors: Mapping[str, LogicalTensor]) -> None:
        self._logical_tensors = logical_tensors
        self._free: dict[tuple[str, int], list[BufferAllocation]] = {}
        self._allocated_by_tensor_name: dict[str, BufferAllocation] = {}

    def acquire(
        self,
        rt: RuntimeSession,
        descriptor_tensor: LogicalTensor,
    ) -> BufferAllocation:
        existing = self._allocated_by_tensor_name.get(descriptor_tensor.name)
        if existing is not None:
            return existing
        if descriptor_tensor.memory not in {MemoryClass.FRAME_WORKSPACE, MemoryClass.OP_SCRATCH}:
            allocation = _allocate_replay_descriptor_tensor(rt, descriptor_tensor)
            self._allocated_by_tensor_name[descriptor_tensor.name] = allocation
            return allocation
        key = _replay_workspace_key(descriptor_tensor)
        bucket = self._free.get(key)
        if bucket:
            allocation = bucket.pop()
        else:
            allocation = _allocate_replay_descriptor_tensor(rt, descriptor_tensor)
        self._allocated_by_tensor_name[descriptor_tensor.name] = allocation
        return allocation

    def recycle_after_dispatch(
        self,
        dispatch_index: int,
        tensor_last_use: Mapping[str, int],
    ) -> None:
        for tensor_name, allocation in tuple(self._allocated_by_tensor_name.items()):
            if tensor_last_use.get(tensor_name) != dispatch_index:
                continue
            del self._allocated_by_tensor_name[tensor_name]
            tensor = self._logical_tensors.get(tensor_name)
            if tensor is None or tensor.memory not in {MemoryClass.FRAME_WORKSPACE, MemoryClass.OP_SCRATCH}:
                continue
            self._free.setdefault(_replay_workspace_key(tensor), []).append(allocation)


def _replay_tensor_last_use(
    entries: tuple[ReplayDispatchTemplate, ...],
    *,
    logical_tensors: Mapping[str, LogicalTensor],
) -> dict[str, int]:
    last_use: dict[str, int] = {}
    for index, entry in enumerate(entries):
        for _field, tensor_name in (*entry.logical_reads, *entry.logical_writes):
            tensor = logical_tensors[tensor_name]
            descriptor_tensor = canonical_replay_descriptor_tensor(
                tensor=tensor,
                logical_tensors=logical_tensors,
            )
            last_use[descriptor_tensor.name] = index
    return last_use


def _replay_workspace_key(tensor: LogicalTensor) -> tuple[str, int]:
    return tensor.spec.dtype, tensor_nbytes(tensor.spec)
