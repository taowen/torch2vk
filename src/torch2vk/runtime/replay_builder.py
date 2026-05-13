"""Replay plan construction and descriptor rebinding for RuntimeSession."""

from __future__ import annotations

import hashlib
import pickle
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
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
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
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
from torch2vk.runtime.shader import (
    DispatchRecord,
    ExprDim,
    IOKind,
    ParamsBufferFieldSpec,
    ParamsBufferSpec,
    PushConstantFieldSpec,
    PushConstantInput,
    PushConstantSpec,
    PushConstantType,
    PushConstantValue,
    ShaderContract,
    ShaderVariant,
    TensorFieldSpec,
    referenced_symbols,
)
from torch2vk.vulkan.allocation import BufferAllocation, BufferSlice
from torch2vk.vulkan.compute_pipeline import BoundComputeBinding, DescriptorBufferBinding
from torch2vk.vulkan.types import tensor_layout_symbol_names, tensor_nbytes

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


_REPLAY_TEMPLATE_CACHE: dict[str, list[ReplayPlanTemplate]] = {}
_REPLAY_TEMPLATE_CACHE_DIR = "replay_templates"


def build_replay_plan(
    rt: RuntimeSession,
    *,
    name: str,
    frame: str,
    readback_tensors: Mapping[str, LogicalTensor] | None = None,
) -> ReplayPlan:
    """Build a ReplayPlan from previously recorded dispatch information."""
    rt._require_open()
    frame_dispatch_records = _frame_dispatch_records(rt, frame)
    num_dispatches = len(frame_dispatch_records)
    if num_dispatches == 0:
        raise ValueError(f"Cannot build replay plan for frame {frame!r} with zero dispatches")
    logical_tensors = _collect_replay_tensors(rt, frame_dispatch_records)
    template = _build_replay_template_from_records(
        rt,
        name=name,
        frame_dispatch_records=frame_dispatch_records,
        logical_tensors=logical_tensors,
    )
    return _instantiate_replay_template(
        rt,
        template=template,
        logical_tensors=logical_tensors,
        readback_tensors=readback_tensors,
    )


def _build_replay_template_from_records(
    rt: RuntimeSession,
    *,
    name: str,
    frame_dispatch_records: Sequence[DispatchRecord],
    logical_tensors: Mapping[str, LogicalTensor],
) -> ReplayPlanTemplate:
    entry_dynamic_symbol_names = tuple(
        _entry_dynamic_symbol_names(
            variant=rt._model_shader(record.shader),
            record=record,
            logical_tensors=logical_tensors,
        )
        for record in frame_dispatch_records
    )
    dynamic_symbol_names = tuple(
        sorted(
            {
                symbol
                for entry_symbols in entry_dynamic_symbol_names
                for symbol in entry_symbols
            }
        )
    )
    entries = tuple(
        ReplayDispatchTemplate(
            shader=record.shader,
            logical_reads=tuple(record.logical_reads),
            logical_writes=tuple(record.logical_writes),
            symbols=tuple(record.symbols),
            dispatch_size=record.dispatch_size,
            dynamic_symbol_names=entry_dynamic_symbol_names[i],
            source_dispatch_index=record.index,
            source_frame=record.frame,
        )
        for i, record in enumerate(frame_dispatch_records)
    )
    return ReplayPlanTemplate(
        name=name,
        entries=entries,
        dynamic_symbol_names=dynamic_symbol_names,
    )


def _instantiate_replay_template(
    rt: RuntimeSession,
    *,
    template: ReplayPlanTemplate,
    logical_tensors: Mapping[str, LogicalTensor],
    readback_tensors: Mapping[str, LogicalTensor] | None = None,
) -> ReplayPlan:
    num_dispatches = len(template.entries)
    dynamic_symbol_names = template.dynamic_symbol_names

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

    for i, template_entry in enumerate(template.entries):
        source_variant = rt._model_shader(template_entry.shader)
        dynamic_symbols = template_entry.dynamic_symbol_names
        variant = _replay_variant_for_dynamic_symbols(source_variant, dynamic_symbols)
        contract = variant.contract
        pipeline = rt._pipeline_for_variant(variant)
        record_symbols = dict(template_entry.symbols)
        logical_by_field = dict(template_entry.logical_reads)
        logical_by_field.update(template_entry.logical_writes)
        tensors = {
            field.name: logical_tensors[logical_by_field[field.name]]
            for field in contract.fields
        }

        buffer_views: list[DescriptorBufferBinding] = []
        descriptor_bindings: list[ReplayDescriptorBinding] = []
        for field in contract.fields:
            tensor = tensors[field.name]
            descriptor_tensor = _canonical_replay_descriptor_tensor(
                tensor=tensor,
                logical_tensors=logical_tensors,
            )
            descriptor_rebindable = _replay_descriptor_rebindable(descriptor_tensor)
            if not _has_live_buffer(descriptor_tensor):
                if descriptor_tensor.memory is MemoryClass.MODEL_WEIGHT:
                    rt._materialize_weight(descriptor_tensor)
                else:
                    alloc = _allocate_replay_descriptor_tensor(
                        rt,
                        descriptor_tensor,
                    )
                    with descriptor_tensor.runtime_write_scope():
                        descriptor_tensor.buffer = BufferSlice(
                            allocation=alloc,
                            offset=alloc.offset,
                            nbytes=tensor_nbytes(descriptor_tensor.spec),
                        )
                        descriptor_tensor.descriptor_nbytes = descriptor_tensor.buffer.nbytes
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
        )

        entry = ReplayDispatchEntry(
            pipeline=pipeline,
            binding=binding,
            descriptors=tuple(descriptor_bindings),
            push_constants=push_constants,
            dispatch_size=template_entry.dispatch_size,
            dispatch_formula=contract.dispatch,
            symbols=record_symbols,
            dynamic_symbol_names=dynamic_symbols,
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
                raise RuntimeError(f"Replay entry for {entry.pipeline.debug_name} has no indirect offset")
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
        template=template,
        readback_slots=readback_slots,
        workspace_allocations=workspace_allocations + params_allocations,
        bindings=bindings,
        profile_state=profile_state,
        profile_recorder=rt.profiler,
    )


def _frame_dispatch_records(
    rt: RuntimeSession,
    frame: str,
) -> Sequence[DispatchRecord]:
    context = rt._frame_history.get(frame)
    if context is None:
        raise KeyError(f"Replay frame {frame!r} was not recorded")
    if context.end_dispatch_index < context.start_dispatch_index:
        raise RuntimeError(
            f"Replay frame {frame!r} has invalid dispatch range "
            f"{context.start_dispatch_index}:{context.end_dispatch_index}"
        )
    return rt.dispatch_records[context.start_dispatch_index:context.end_dispatch_index]


def _entry_dynamic_symbol_names(
    *,
    variant: ShaderVariant,
    record: DispatchRecord,
    logical_tensors: Mapping[str, LogicalTensor],
) -> tuple[str, ...]:
    record_symbols = set(dict(record.symbols))
    logical_by_field = dict(record.logical_reads)
    logical_by_field.update(record.logical_writes)
    rebindable_symbols: set[str] = set()
    static_symbols: set[str] = set()
    for field in variant.contract.fields:
        tensor = logical_tensors[logical_by_field[field.name]]
        descriptor_tensor = _canonical_replay_descriptor_tensor(
            tensor=tensor,
            logical_tensors=logical_tensors,
        )
        field_symbols = set(_tensor_field_symbol_names(field))
        if _replay_descriptor_rebindable(descriptor_tensor):
            rebindable_symbols.update(field_symbols)
            continue
        static_symbols.update(field_symbols)
    dynamic_symbols = rebindable_symbols - static_symbols
    return tuple(sorted(symbol for symbol in dynamic_symbols if symbol in record_symbols))


def _tensor_field_symbol_names(field: TensorFieldSpec) -> tuple[str, ...]:
    symbols = list(_referenced_symbols_in_dims(field.contract.shape))
    symbols.extend(tensor_layout_symbol_names(field.contract.layout))
    return tuple(symbols)


def _referenced_symbols_in_dims(values: Sequence[ExprDim]) -> tuple[str, ...]:
    symbols: list[str] = []
    for value in values:
        symbols.extend(referenced_symbols(value))
    return tuple(symbols)


_PUSH_CONSTANT_DECL_RE = re.compile(
    r"layout\s*\(\s*push_constant\s*\)\s*uniform\s+\w+\s*\{.*?\}\s*pc\s*;",
    re.DOTALL,
)


def _replay_variant_for_dynamic_symbols(
    variant: ShaderVariant,
    dynamic_symbol_names: tuple[str, ...],
) -> ShaderVariant:
    push_constants = variant.contract.push_constants
    if push_constants is None or not dynamic_symbol_names:
        return variant

    dynamic_symbols = set(dynamic_symbol_names)
    dynamic_fields = tuple(
        field
        for field in push_constants.fields
        if dynamic_symbols.intersection(_push_constant_value_symbols(field.value))
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
    if isinstance(field.value, PushConstantInput):
        raise RuntimeError(
            f"Replay shader {shader_name!r} cannot move PushConstantInput "
            f"{field.name!r} into params"
        )
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


def _collect_replay_tensors(
    rt: RuntimeSession,
    records: Sequence[DispatchRecord],
) -> dict[str, LogicalTensor]:
    logical_tensors = rt._named_model_tensors()
    frame_names = {record.frame for record in records}

    def collect(tensor: LogicalTensor) -> None:
        existing = logical_tensors.get(tensor.name)
        if existing is not None and existing is not tensor:
            raise RuntimeError(
                f"Replay record has multiple LogicalTensor objects named {tensor.name!r}"
            )
        logical_tensors[tensor.name] = tensor

    for record in records:
        for _, tensor in (*record.reads, *record.writes):
            collect(tensor)
    for frame_name in frame_names:
        frame = rt._frame_history.get(frame_name)
        if frame is None:
            continue
        for tensor in (*frame.registered_inputs, *frame.used_tensors, *frame.written_tensors):
            collect(tensor)
    return logical_tensors


def rebind_replay_plan(rt: RuntimeSession, plan: ReplayPlan) -> None:
    """Retarget replay descriptors to a compatible tensor set without recording."""
    rt._require_open()
    if plan._closed:
        raise RuntimeError(f"ReplayPlan {plan.name!r} is closed")
    if plan.device is not rt.device:
        raise ValueError("ReplayPlan belongs to a different RuntimeSession device")
    if plan.readback_slots:
        raise RuntimeError(
            "Replay plans with baked readback copy commands cannot be rebound"
        )

    logical_tensors = rt._named_model_tensors()
    for entry in plan.dispatch_entries:
        rebound_descriptors: list[ReplayDescriptorBinding] = []
        shape_field_tensors: dict[str, LogicalTensor] = {}
        descriptors_changed = False
        for descriptor in entry.descriptors:
            if not descriptor.rebindable:
                rebound_descriptors.append(descriptor)
                continue
            try:
                tensor = logical_tensors[descriptor.tensor_name]
            except KeyError as exc:
                raise KeyError(
                    f"ReplayPlan {plan.name!r} rebind is missing tensor "
                    f"{descriptor.tensor_name!r}"
                ) from exc
            _materialize_replay_rebind_tensor(rt, tensor, field=descriptor.field)
            if tensor.buffer is None:
                raise RuntimeError(f"{tensor.name} is not materialized for replay rebind")
            if descriptor.validate_shape:
                shape_field_tensors[descriptor.field.name] = tensor
            if descriptor.buffer.matches_slice(
                tensor.buffer,
                descriptor_nbytes=tensor.descriptor_nbytes,
            ):
                rebound_descriptors.append(descriptor)
                continue
            rebound_descriptors.append(
                replace(
                    descriptor,
                    buffer=DescriptorBufferBinding.from_slice(
                        tensor.buffer,
                        descriptor_nbytes=tensor.descriptor_nbytes,
                    ),
                )
            )
            descriptors_changed = True

        if shape_field_tensors:
            rebound_symbols = _validate_replay_rebind_symbols(
                rt,
                plan=plan,
                entry=entry,
                field_tensors=shape_field_tensors,
            )
            if rebound_symbols:
                updated_symbols = dict(entry.symbols)
                for symbol_name in entry.dynamic_symbol_names:
                    rebound_value = rebound_symbols.get(symbol_name)
                    if rebound_value is not None:
                        updated_symbols[symbol_name] = rebound_value
                entry.symbols = updated_symbols

        if not descriptors_changed:
            continue

        buffer_views = [descriptor.buffer for descriptor in rebound_descriptors]
        if entry.params_buffer is not None:
            if entry.params_layout is None:
                raise RuntimeError("Replay params entry is missing its params layout")
            buffer_views.append(
                DescriptorBufferBinding(
                    slice=BufferSlice(
                        allocation=entry.params_buffer,
                        offset=entry.params_buffer.offset,
                        nbytes=entry.params_layout.size,
                    )
                )
            )
        entry.pipeline.update_bound_buffers(entry.binding, buffer_views)
        entry.descriptors = tuple(rebound_descriptors)


def replay_plan_compatible(rt: RuntimeSession, plan: ReplayPlan) -> bool:
    rt._require_open()
    if plan._closed or plan.device is not rt.device:
        return False
    logical_tensors = rt._named_model_tensors()
    for entry in plan.dispatch_entries:
        field_tensors: dict[str, LogicalTensor] = {}
        for descriptor in entry.descriptors:
            if not descriptor.rebindable:
                continue
            tensor = logical_tensors.get(descriptor.tensor_name)
            if tensor is None:
                return False
            if descriptor.validate_shape:
                field_tensors[descriptor.field.name] = tensor
        try:
            _validate_replay_rebind_symbols(
                rt,
                plan=plan,
                entry=entry,
                field_tensors=field_tensors,
            )
        except ValueError:
            return False
    return True


def cached_replay_plans(rt: RuntimeSession, namespace: str) -> tuple[ReplayPlan, ...]:
    rt._require_open()
    plans = rt._replay_plan_cache.get(namespace, [])
    live_plans = [
        plan
        for plan in plans
        if not plan._closed and replay_plan_compatible(rt, plan)
    ]
    templates = _cached_replay_templates(rt, namespace)
    for template in templates:
        if any(plan.template == template for plan in live_plans):
            continue
        if not _replay_template_compatible(rt, template):
            continue
        live_plans.append(
            _instantiate_replay_template(
                rt,
                template=template,
                logical_tensors=rt._named_model_tensors(),
            )
        )
    rt._replay_plan_cache[namespace] = live_plans
    if not live_plans and (
        any(not plan._closed for plan in plans) or templates
    ):
        raise RuntimeError(
            f"Replay cache {namespace!r} exists but is incompatible with current model tensors"
        )
    return tuple(live_plans)


def cache_replay_plan(rt: RuntimeSession, namespace: str, plan: ReplayPlan) -> None:
    rt._require_open()
    if plan._closed:
        raise RuntimeError(f"Cannot cache closed ReplayPlan {plan.name!r}")
    if plan.device is not rt.device:
        raise ValueError("ReplayPlan belongs to a different RuntimeSession device")
    plans = rt._replay_plan_cache.setdefault(namespace, [])
    if not any(existing is plan for existing in plans):
        plans.append(plan)
    if plan.template is not None:
        templates = _cached_replay_templates(rt, namespace)
        if plan.template not in templates:
            templates.append(plan.template)
            _write_replay_templates(rt, namespace, templates)


def _cached_replay_templates(
    rt: RuntimeSession,
    namespace: str,
) -> list[ReplayPlanTemplate]:
    cached = _REPLAY_TEMPLATE_CACHE.get(namespace)
    if cached is not None:
        return cached
    path = _replay_template_cache_path(rt, namespace)
    if not path.is_file():
        _REPLAY_TEMPLATE_CACHE[namespace] = []
        return _REPLAY_TEMPLATE_CACHE[namespace]
    with path.open("rb") as handle:
        loaded: object = pickle.load(handle)
    if not isinstance(loaded, list):
        raise TypeError(f"Replay template cache {path} did not contain a list")
    templates: list[ReplayPlanTemplate] = []
    for item in loaded:
        if not isinstance(item, ReplayPlanTemplate):
            raise TypeError(
                f"Replay template cache {path} contained {type(item).__name__}, "
                "expected ReplayPlanTemplate"
            )
        templates.append(item)
    _REPLAY_TEMPLATE_CACHE[namespace] = templates
    return templates


def _write_replay_templates(
    rt: RuntimeSession,
    namespace: str,
    templates: Sequence[ReplayPlanTemplate],
) -> None:
    path = _replay_template_cache_path(rt, namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(list(templates), handle, protocol=pickle.HIGHEST_PROTOCOL)


def _replay_template_cache_path(rt: RuntimeSession, namespace: str) -> Path:
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    return rt.artifact_dir.parent / _REPLAY_TEMPLATE_CACHE_DIR / f"{digest}.pkl"


def _replay_template_compatible(rt: RuntimeSession, template: ReplayPlanTemplate) -> bool:
    logical_tensors = rt._named_model_tensors()
    for entry in template.entries:
        source_variant = rt._model_shader(entry.shader)
        logical_by_field = dict(entry.logical_reads)
        logical_by_field.update(entry.logical_writes)
        field_tensors: dict[str, LogicalTensor] = {}
        for field in source_variant.contract.fields:
            tensor_name = logical_by_field.get(field.name)
            if tensor_name is None:
                return False
            tensor = logical_tensors.get(tensor_name)
            if tensor is None:
                return False
            descriptor_tensor = _canonical_replay_descriptor_tensor(
                tensor=tensor,
                logical_tensors=logical_tensors,
            )
            if (
                _replay_descriptor_rebindable(descriptor_tensor)
                and descriptor_tensor is tensor
            ):
                field_tensors[field.name] = tensor
        try:
            rebound_symbols = rt._bind_shape_symbols(
                tuple(
                    field
                    for field in source_variant.contract.fields
                    if field.name in field_tensors
                ),
                field_tensors,
            )
        except ValueError:
            return False
        if not _replay_symbols_compatible(
            plan_name=template.name,
            entry_symbols=dict(entry.symbols),
            dynamic_symbol_names=entry.dynamic_symbol_names,
            rebound_symbols=rebound_symbols,
        ):
            return False
    return True


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
        f"{descriptor_tensor.name} is not materialized and cannot be "
        "allocated as replay workspace"
    )


def _materialize_replay_rebind_tensor(
    rt: RuntimeSession,
    tensor: LogicalTensor,
    *,
    field: TensorFieldSpec,
) -> None:
    tensor.validate_declaration()
    if _has_live_buffer(tensor):
        return
    if tensor.buffer is not None:
        with tensor.runtime_write_scope():
            tensor.buffer = None
            tensor.descriptor_nbytes = None
    if field.io_kind is IOKind.INPUT:
        rt._materialize_read(tensor)
        return
    if field.io_kind is IOKind.OUTPUT:
        rt._materialize_write(tensor, io_kind=field.io_kind)
        return
    rt._materialize_read(tensor)


def _canonical_replay_descriptor_tensor(
    *,
    tensor: LogicalTensor,
    logical_tensors: Mapping[str, LogicalTensor],
) -> LogicalTensor:
    if tensor.memory is not MemoryClass.FRAME_WORKSPACE:
        return tensor
    alias_source = tensor.alias_source
    if alias_source is not None:
        return _canonical_replay_descriptor_tensor(
            tensor=alias_source,
            logical_tensors=logical_tensors,
        )
    alias_owner = _live_non_frame_alias_owner(tensor, logical_tensors)
    if alias_owner is not None:
        return alias_owner
    return tensor


def _replay_descriptor_rebindable(tensor: LogicalTensor) -> bool:
    return tensor.memory not in {
        MemoryClass.FRAME_WORKSPACE,
        MemoryClass.MODEL_WEIGHT,
        MemoryClass.OP_SCRATCH,
    }


def _has_live_buffer(tensor: LogicalTensor) -> bool:
    return tensor.buffer is not None and not tensor.buffer.allocation.released


def _live_non_frame_alias_owner(
    tensor: LogicalTensor,
    logical_tensors: Mapping[str, LogicalTensor],
) -> LogicalTensor | None:
    if not _has_live_buffer(tensor):
        return None
    for candidate in logical_tensors.values():
        if candidate is tensor or candidate.memory is MemoryClass.FRAME_WORKSPACE:
            continue
        if _has_live_buffer(candidate) and candidate.buffer == tensor.buffer:
            return candidate
    return None


def _validate_replay_rebind_symbols(
    rt: RuntimeSession,
    *,
    plan: ReplayPlan,
    entry: ReplayDispatchEntry,
    field_tensors: Mapping[str, LogicalTensor],
) -> dict[str, int]:
    if not field_tensors:
        return {}
    rebound_symbols = rt._bind_shape_symbols(
        tuple(
            descriptor.field
            for descriptor in entry.descriptors
            if descriptor.validate_shape and descriptor.field.name in field_tensors
        ),
        field_tensors,
    )
    if not _replay_symbols_compatible(
        plan_name=plan.name,
        entry_symbols=entry.symbols,
        dynamic_symbol_names=entry.dynamic_symbol_names,
        rebound_symbols=rebound_symbols,
    ):
        _raise_replay_symbol_mismatch(
            plan_name=plan.name,
            entry_symbols=entry.symbols,
            dynamic_symbol_names=entry.dynamic_symbol_names,
            rebound_symbols=rebound_symbols,
        )
    return rebound_symbols


def _replay_symbols_compatible(
    *,
    plan_name: str,
    entry_symbols: Mapping[str, int],
    dynamic_symbol_names: tuple[str, ...],
    rebound_symbols: Mapping[str, int],
) -> bool:
    dynamic_symbols = set(dynamic_symbol_names)
    for name, rebound_value in rebound_symbols.items():
        if name in dynamic_symbols:
            continue
        original_value = entry_symbols.get(name)
        if original_value is not None and rebound_value != original_value:
            return False
    return True


def _raise_replay_symbol_mismatch(
    *,
    plan_name: str,
    entry_symbols: Mapping[str, int],
    dynamic_symbol_names: tuple[str, ...],
    rebound_symbols: Mapping[str, int],
) -> None:
    dynamic_symbols = set(dynamic_symbol_names)
    for name, rebound_value in rebound_symbols.items():
        if name in dynamic_symbols:
            continue
        original_value = entry_symbols.get(name)
        if original_value is not None and rebound_value != original_value:
            raise ValueError(
                f"ReplayPlan {plan_name!r} cannot rebind static symbol {name!r}: "
                f"recorded {original_value}, got {rebound_value}"
            )
