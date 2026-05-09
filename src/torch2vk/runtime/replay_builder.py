"""Replay plan construction and descriptor rebinding for RuntimeSession."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
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

from torch2vk.runtime.logical import LogicalTensor, MemoryClass
from torch2vk.runtime.replay import (
    ReplayDescriptorBinding,
    ReplayDispatchEntry,
    ReplayPlan,
    ReplayProfileState,
    ReplayReadbackSlot,
)
from torch2vk.runtime.shader import DispatchRecord, IOKind, ShaderVariant, TensorFieldSpec
from torch2vk.vulkan.allocation import BufferAllocation, BufferSlice
from torch2vk.vulkan.compute_pipeline import BoundComputeBinding, DescriptorBufferBinding
from torch2vk.vulkan.types import tensor_nbytes

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


def build_replay_plan(
    rt: RuntimeSession,
    *,
    name: str,
    frame_dispatch_records: Sequence[DispatchRecord],
    dynamic_symbol_names: tuple[str, ...] = (),
    readback_tensors: Mapping[str, LogicalTensor] | None = None,
    token_feedback_source: LogicalTensor | None = None,
    token_feedback_target: LogicalTensor | None = None,
) -> ReplayPlan:
    """Build a ReplayPlan from previously recorded dispatch information."""
    rt._require_open()
    num_dispatches = len(frame_dispatch_records)
    if num_dispatches == 0:
        raise ValueError("Cannot build replay plan with zero dispatches")
    model_shaders = rt._named_model_shaders()

    if (token_feedback_source is None) != (token_feedback_target is None):
        raise ValueError(
            "token_feedback_source and token_feedback_target must be provided together"
        )
    if token_feedback_source is not None and token_feedback_target is not None:
        _validate_token_feedback_tensors(
            token_feedback_source,
            token_feedback_target,
        )
    logical_tensors = _collect_replay_tensors(rt, frame_dispatch_records)

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

    for i, record in enumerate(frame_dispatch_records):
        variant = model_shaders[record.shader]
        contract = variant.contract
        pipeline = rt._pipeline_for_variant(variant)
        record_symbols = dict(record.symbols)
        logical_by_field = dict(record.logical_reads)
        logical_by_field.update(record.logical_writes)
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
            if (
                token_feedback_source is not None
                and token_feedback_target is not None
                and tensor.name == token_feedback_target.name
            ):
                descriptor_tensor = token_feedback_source
            descriptor_rebindable = descriptor_tensor.memory not in {
                MemoryClass.FRAME_WORKSPACE,
                MemoryClass.MODEL_WEIGHT,
                MemoryClass.OP_SCRATCH,
            }
            if not _has_live_buffer(descriptor_tensor):
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
            dispatch_size=record.dispatch_size,
            dispatch_formula=contract.dispatch,
            symbols=record_symbols,
            indirect_offset=i * 12 if use_indirect_dispatch else None,
            params_buffer=params_alloc,
            params_layout=contract.params_buffer,
            source_dispatch_index=record.index,
            source_frame=record.frame,
            source_shader=record.shader,
            source_logical_reads=tuple(record.logical_reads),
            source_logical_writes=tuple(record.logical_writes),
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
        name=name,
        command_buffer=command_buffer,
        fence=fence,
        indirect_buffer=indirect_buffer,
        num_dispatches=num_dispatches,
        dispatch_entries=tuple(dispatch_entries),
        params_entries=tuple(params_entries),
        dynamic_symbol_names=dynamic_symbol_names,
        readback_slots=readback_slots,
        workspace_allocations=workspace_allocations + params_allocations,
        bindings=bindings,
        profile_state=profile_state,
        profile_recorder=rt.profiler,
    )


def _validate_token_feedback_tensors(
    source: LogicalTensor,
    target: LogicalTensor,
) -> None:
    if source.spec.dtype != target.spec.dtype:
        raise ValueError(
            "Replay token feedback requires matching dtypes, got "
            f"{source.name}={source.spec.dtype} and {target.name}={target.spec.dtype}"
        )
    source_nbytes = tensor_nbytes(source.spec)
    target_nbytes = tensor_nbytes(target.spec)
    if source_nbytes != target_nbytes:
        raise ValueError(
            "Replay token feedback requires matching byte sizes, got "
            f"{source.name}={source_nbytes} bytes and {target.name}={target_nbytes} bytes"
        )


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
        rebound_field_tensors: dict[str, LogicalTensor] = {}
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
            binding = DescriptorBufferBinding.from_slice(
                tensor.buffer,
                descriptor_nbytes=tensor.descriptor_nbytes,
            )
            rebound_descriptors.append(replace(descriptor, buffer=binding))
            rebound_field_tensors[descriptor.field.name] = tensor

        if rebound_field_tensors:
            _validate_replay_rebind_symbols(
                rt,
                plan=plan,
                entry=entry,
                field_tensors={
                    descriptor.field.name: rebound_field_tensors[descriptor.field.name]
                    for descriptor in rebound_descriptors
                    if descriptor.validate_shape
                    and descriptor.field.name in rebound_field_tensors
                },
            )

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
    live_plans = [plan for plan in plans if not plan._closed]
    if len(live_plans) != len(plans):
        rt._replay_plan_cache[namespace] = live_plans
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
        if descriptor_tensor not in rt._inputs:
            raise RuntimeError(f"{descriptor_tensor.name} requires missing replay input")
        alloc = rt.device.allocate_host_visible_allocation(nbytes)
        alloc.buffer.map_persistent()
        array = np.ascontiguousarray(rt._inputs[descriptor_tensor])
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
) -> None:
    if not field_tensors:
        return
    rebound_symbols = rt._bind_shape_symbols(
        tuple(
            descriptor.field
            for descriptor in entry.descriptors
            if descriptor.validate_shape and descriptor.field.name in field_tensors
        ),
        field_tensors,
    )
    dynamic_symbols = set(plan.dynamic_symbol_names)
    for name, rebound_value in rebound_symbols.items():
        if name in dynamic_symbols:
            continue
        original_value = entry.symbols.get(name)
        if original_value is not None and rebound_value != original_value:
            raise ValueError(
                f"ReplayPlan {plan.name!r} cannot rebind static symbol {name!r}: "
                f"recorded {original_value}, got {rebound_value}"
            )
