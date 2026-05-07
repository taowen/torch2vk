"""Tensor scaffold inference from FX nodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch2vk.exportv2.fx import FxNodeProjector, StaticNode, project_fx_node, project_fx_nodes
from torch2vk.exportv2.lowering import DEFAULT_LOWERING_REGISTRY, OpLoweringRegistry
from torch2vk.exportv2.protocols import FxNodeLike
from torch2vk.exportv2.tensor_codegen import (
    tensor_scaffold_fields_from_static_nodes as tensor_scaffold_fields_from_codegen_nodes,
)
from torch2vk.exportv2.tensor_pattern import TensorFieldPattern
from torch2vk.runtime.shader import ShaderVariant


def tensor_scaffold_fields_from_fx_nodes(
    nodes: Sequence[FxNodeLike],
    *,
    shader_variants: Mapping[str, ShaderVariant],
    name_map: Mapping[str, str] | None = None,
    project: FxNodeProjector = project_fx_node,
    parameter_sources: Mapping[str, str] | None = None,
    extra_fields: Sequence[TensorFieldPattern] = (),
    external_fields: Sequence[str] = (),
    role_overrides: Mapping[str, str] | None = None,
    dtype_overrides: Mapping[str, str] | None = None,
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
    include_unresolved_ops: bool = True,
) -> tuple[TensorFieldPattern, ...]:
    return tensor_scaffold_fields_from_static_nodes(
        project_fx_nodes(nodes, name_map=name_map, project=project),
        shader_variants=shader_variants,
        parameter_sources=parameter_sources,
        extra_fields=extra_fields,
        external_fields=external_fields,
        role_overrides=role_overrides,
        dtype_overrides=dtype_overrides,
        lowering_registry=lowering_registry,
        include_unresolved_ops=include_unresolved_ops,
    )


def tensor_scaffold_fields_from_static_nodes(
    nodes: Sequence[StaticNode],
    *,
    shader_variants: Mapping[str, ShaderVariant],
    parameter_sources: Mapping[str, str] | None = None,
    extra_fields: Sequence[TensorFieldPattern] = (),
    external_fields: Sequence[str] = (),
    role_overrides: Mapping[str, str] | None = None,
    dtype_overrides: Mapping[str, str] | None = None,
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
    include_unresolved_ops: bool = True,
) -> tuple[TensorFieldPattern, ...]:
    return tensor_scaffold_fields_from_codegen_nodes(
        nodes=nodes,
        shader_variants=shader_variants,
        parameter_sources=parameter_sources,
        extra_fields=extra_fields,
        external_fields=external_fields,
        role_overrides=role_overrides,
        dtype_overrides=dtype_overrides,
        lowering_registry=lowering_registry,
        include_unresolved_ops=include_unresolved_ops,
    )
