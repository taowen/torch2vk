"""Tensor scaffold helpers for reflected PyTorch modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Protocol, TypeVar

from torch2vk.exportv2.fx import StaticNode
from torch2vk.exportv2.reflection import TorchModuleReflection
from torch2vk.exportv2.tensor_pattern import TensorFieldPattern
from torch2vk.exportv2.tensor_scaffold import tensor_scaffold_fields_from_static_nodes
from torch2vk.runtime.shader import ShaderVariant


class TensorFieldLike(Protocol):
    @property
    def field(self) -> str: ...


TensorFieldT = TypeVar("TensorFieldT", bound=TensorFieldLike)


def tensor_fields_from_reflected_static_nodes(
    *,
    reflection: TorchModuleReflection,
    module_prefix: str,
    nodes: Sequence[StaticNode],
    shader_variants: Mapping[str, ShaderVariant],
    relative_parameter_sources: bool,
    extra_fields: Sequence[TensorFieldPattern] = (),
    external_fields: Sequence[str] = (),
    role_overrides: Mapping[str, str] | None = None,
    dtype_overrides: Mapping[str, str] | None = None,
    include_unresolved_ops: bool = True,
    field_order: Sequence[str] = (),
) -> tuple[TensorFieldPattern, ...]:
    fields = tensor_scaffold_fields_from_static_nodes(
        nodes,
        shader_variants=shader_variants,
        extra_fields=extra_fields,
        external_fields=external_fields,
        role_overrides=role_overrides,
        dtype_overrides=dtype_overrides,
        include_unresolved_ops=include_unresolved_ops,
    )
    fields = attach_parameter_sources(
        fields,
        parameter_sources_from_reflection(
            reflection,
            module_prefix=module_prefix,
            fields=fields,
            relative_to_module=relative_parameter_sources,
        ),
    )
    return order_tensor_fields(fields, field_order) if field_order else fields


def attach_parameter_sources(
    fields: Sequence[TensorFieldPattern],
    sources: Mapping[str, str],
) -> tuple[TensorFieldPattern, ...]:
    return tuple(
        replace(field, source_parameter=sources[field.field]) if field.field in sources else field
        for field in fields
    )


def parameter_sources_from_reflection(
    reflection: TorchModuleReflection,
    *,
    module_prefix: str,
    fields: Sequence[TensorFieldPattern],
    relative_to_module: bool,
) -> dict[str, str]:
    prefix = f"{module_prefix}."
    wanted = {field.field for field in fields}
    sources: dict[str, str] = {}
    for parameter_name in sorted(reflection.parameter_shapes):
        if not parameter_name.startswith(prefix):
            continue
        relative_name = parameter_name[len(prefix) :]
        field = parameter_field_name(relative_name)
        if field in wanted:
            sources[field] = relative_name if relative_to_module else parameter_name
    return sources


def parameter_field_name(relative_parameter: str) -> str:
    parts = relative_parameter.split(".")
    if len(parts) < 2:
        raise ValueError(f"parameter path must include a module and leaf: {relative_parameter!r}")
    return f"{parts[-2]}_{parts[-1]}"


def order_tensor_fields(
    fields: Sequence[TensorFieldT],
    order: Sequence[str],
) -> tuple[TensorFieldT, ...]:
    by_name = {field.field: field for field in fields}
    ordered = [by_name.pop(name) for name in order if name in by_name]
    ordered.extend(by_name.values())
    return tuple(ordered)
