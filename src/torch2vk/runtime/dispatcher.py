"""Shader dispatch execution for RuntimeSession."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from torch2vk.runtime.logical import DispatchWriter, LogicalTensor
from torch2vk.runtime.materialization import (
    descriptor_view_for_field,
    record_descriptor_view,
    record_tensor_snapshot,
)
from torch2vk.runtime.shader import DispatchRecord, ShaderVariant, eval_expr
from torch2vk.vulkan.allocation import BufferAllocation, BufferSlice
from torch2vk.vulkan.compute_pipeline import DescriptorBufferBinding

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


def dispatch(rt: RuntimeSession, variant: ShaderVariant, **arguments: object) -> None:
    rt._require_open()
    frame = rt._current_frame()
    contract = variant.contract
    expected = {field.name for field in contract.fields}
    provided = set(arguments)
    if missing := expected - provided:
        raise ValueError(f"{variant.name} missing tensor fields: {sorted(missing)}")
    if extra := provided - expected:
        raise ValueError(f"{variant.name} got unexpected fields: {sorted(extra)}")

    tensors: dict[str, LogicalTensor] = {}
    for name, argument in arguments.items():
        if not isinstance(argument, LogicalTensor):
            raise TypeError(
                f"{variant.name}.{name} expects LogicalTensor, got {type(argument).__name__}"
            )
        argument.validate_declaration()
        tensors[name] = argument
    frame.used_tensors.extend(tensors[field.name] for field in contract.fields)

    symbols = rt._bind_shape_symbols(contract.fields, tensors)
    for field in contract.input_fields:
        rt._materialize_read(tensors[field.name])
    for field in contract.output_fields:
        rt._materialize_write(tensors[field.name], io_kind=field.io_kind)

    descriptor_views = tuple(
        descriptor_view_for_field(field, tensors[field.name]) for field in contract.fields
    )
    params_allocation: BufferAllocation | None = None
    if contract.params_buffer is not None:
        params_allocation = rt._materialize_params_buffer(
            contract.params_buffer,
            tensors=tensors,
            symbols=symbols,
        )
        params_slice = BufferSlice(
            allocation=params_allocation,
            offset=params_allocation.offset,
            nbytes=contract.params_buffer.size,
        )
        descriptor_views = descriptor_views + (
            (
                f"__params_{contract.params_buffer.binding_index}",
                DescriptorBufferBinding(slice=params_slice),
            ),
        )
    push_constants, push_values = rt._pack_push_constants(
        contract.push_constants,
        tensors=tensors,
        symbols=symbols,
    )
    dispatch_size = (
        eval_expr(contract.dispatch[0], symbols),
        eval_expr(contract.dispatch[1], symbols),
        eval_expr(contract.dispatch[2], symbols),
    )
    if any(dim <= 0 for dim in dispatch_size):
        raise ValueError(f"{variant.name} resolved non-positive dispatch {dispatch_size}")

    index = len(rt._dispatch_records)
    pipeline = rt._pipeline_for_variant(variant)
    dispatch_started_ns = time.perf_counter_ns()
    try:
        pipeline.dispatch(
            buffers=[view for _, view in descriptor_views],
            group_count_x=dispatch_size[0],
            group_count_y=dispatch_size[1],
            group_count_z=dispatch_size[2],
            push_constants=push_constants,
            debug_label=rt.profiler.sqtt_label(
                frame=frame.frame,
                shader=variant.name,
                dispatch_index=index,
            ),
        )
        elapsed_wall_ns = time.perf_counter_ns() - dispatch_started_ns
    finally:
        if params_allocation is not None:
            params_allocation.close()

    record = DispatchRecord(
        index=index,
        frame=frame.frame,
        shader=variant.name,
        reads=tuple((field.name, tensors[field.name]) for field in contract.input_fields),
        writes=tuple((field.name, tensors[field.name]) for field in contract.output_fields),
        logical_reads=tuple(
            (field.name, tensors[field.name].name) for field in contract.input_fields
        ),
        logical_writes=tuple(
            (field.name, tensors[field.name].name) for field in contract.output_fields
        ),
        symbols=tuple(sorted(symbols.items())),
        dispatch_size=dispatch_size,
        descriptor_views=tuple(
            record_descriptor_view(index, field, tensors[field.name])
            for index, field in enumerate(contract.fields)
        ),
        tensor_snapshots=tuple(
            record_tensor_snapshot(field, tensors[field.name]) for field in contract.fields
        ),
        push_constant_values=tuple(sorted(push_values.items())),
    )
    rt._dispatch_records.append(record)
    rt.profiler.record_dispatch(
        record=record,
        pipeline=pipeline,
        elapsed_wall_ns=elapsed_wall_ns,
    )
    for field in contract.output_fields:
        tensor = tensors[field.name]
        with tensor.runtime_write_scope():
            tensor.version += 1
            tensor.writer = DispatchWriter(
                frame=frame.frame,
                shader=variant.name,
                dispatch_index=index,
            )
        frame.written_tensors.append(tensor)
