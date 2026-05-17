"""Replay plan descriptor rebinding and compatibility checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.replay import ReplayDescriptorBinding, ReplayDispatchEntry, ReplayPlan
from torch2vk.runtime.replay_descriptor import has_live_buffer
from torch2vk.runtime.shader import IOKind, TensorFieldSpec
from torch2vk.vulkan.allocation import BufferSlice
from torch2vk.vulkan.compute_pipeline import DescriptorBufferBinding

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


def rebind_replay_plan(rt: RuntimeSession, plan: ReplayPlan) -> None:
    """Retarget replay descriptors to a compatible tensor set without recording."""
    rt._require_open()
    if plan._closed:
        raise RuntimeError(f"ReplayPlan {plan.name!r} is closed")
    if plan.device is not rt.device:
        raise ValueError("ReplayPlan belongs to a different RuntimeSession device")
    if plan.readback_slots:
        raise RuntimeError("Replay plans with baked readback copy commands cannot be rebound")
    if plan.in_flight:
        rt.device.wait_pending_submits()

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
                    f"ReplayPlan {plan.name!r} rebind is missing tensor {descriptor.tensor_name!r}"
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
            rebound_symbols = validate_replay_rebind_symbols(
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
            validate_replay_rebind_symbols(
                rt,
                plan=plan,
                entry=entry,
                field_tensors=field_tensors,
            )
        except ValueError:
            return False
    return True


def _materialize_replay_rebind_tensor(
    rt: RuntimeSession,
    tensor: LogicalTensor,
    *,
    field: TensorFieldSpec,
) -> None:
    tensor.validate_declaration()
    if has_live_buffer(tensor):
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


def validate_replay_rebind_symbols(
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
    if not replay_symbols_compatible(
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


def replay_symbols_compatible(
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
