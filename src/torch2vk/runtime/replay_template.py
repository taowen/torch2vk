"""Replay template construction from recorded dispatches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.replay import ReplayDispatchTemplate, ReplayPlanTemplate
from torch2vk.runtime.replay_descriptor import (
    canonical_replay_descriptor_tensor,
    replay_descriptor_rebindable,
)
from torch2vk.runtime.shader import (
    DispatchRecord,
    ExprDim,
    PushConstantInput,
    ShaderVariant,
    TensorFieldSpec,
    referenced_symbols,
)
from torch2vk.vulkan.types import tensor_layout_symbol_names

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


def build_replay_template_from_records(
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
    entry_dynamic_push_constant_names = tuple(
        _entry_dynamic_push_constant_names(rt._model_shader(record.shader))
        for record in frame_dispatch_records
    )
    dynamic_symbol_names = tuple(
        sorted({symbol for entry_symbols in entry_dynamic_symbol_names for symbol in entry_symbols})
    )
    dynamic_push_constant_names = tuple(
        sorted({name for entry_names in entry_dynamic_push_constant_names for name in entry_names})
    )
    entries = tuple(
        ReplayDispatchTemplate(
            shader=record.shader,
            logical_reads=tuple(record.logical_reads),
            logical_writes=tuple(record.logical_writes),
            symbols=tuple(record.symbols),
            dispatch_size=record.dispatch_size,
            push_constant_values=record.push_constant_values,
            dynamic_symbol_names=entry_dynamic_symbol_names[i],
            dynamic_push_constant_names=entry_dynamic_push_constant_names[i],
            source_dispatch_index=record.index,
            source_frame=record.frame,
        )
        for i, record in enumerate(frame_dispatch_records)
    )
    return ReplayPlanTemplate(
        name=name,
        entries=entries,
        dynamic_symbol_names=dynamic_symbol_names,
        dynamic_push_constant_names=dynamic_push_constant_names,
    )


def frame_dispatch_records(
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
    return rt.dispatch_records[context.start_dispatch_index : context.end_dispatch_index]


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
        descriptor_tensor = canonical_replay_descriptor_tensor(
            tensor=tensor,
            logical_tensors=logical_tensors,
        )
        field_symbols = set(_tensor_field_symbol_names(field))
        if replay_descriptor_rebindable(descriptor_tensor):
            rebindable_symbols.update(field_symbols)
            continue
        static_symbols.update(field_symbols)
    dynamic_symbols = rebindable_symbols - static_symbols
    return tuple(sorted(symbol for symbol in dynamic_symbols if symbol in record_symbols))


def _entry_dynamic_push_constant_names(variant: ShaderVariant) -> tuple[str, ...]:
    push_constants = variant.contract.push_constants
    if push_constants is None:
        return ()
    return tuple(
        sorted(
            field.value.name
            for field in push_constants.fields
            if isinstance(field.value, PushConstantInput)
        )
    )


def _tensor_field_symbol_names(field: TensorFieldSpec) -> tuple[str, ...]:
    symbols = list(_referenced_symbols_in_dims(field.contract.shape))
    symbols.extend(tensor_layout_symbol_names(field.contract.layout))
    return tuple(symbols)


def _referenced_symbols_in_dims(values: Sequence[ExprDim]) -> tuple[str, ...]:
    symbols: list[str] = []
    for value in values:
        symbols.extend(referenced_symbols(value))
    return tuple(symbols)


def collect_replay_tensors(
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
