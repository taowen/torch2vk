"""Static dispatch-body generation from FX nodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch2vk.exportv2.lowering import DEFAULT_LOWERING_REGISTRY, OpLoweringRegistry
from torch2vk.exportv2.protocols import FxNodeLike
from torch2vk.exportv2.fx import FxNodeProjector, StaticNode, project_fx_node, project_fx_nodes
from torch2vk.runtime.shader import IOKind, ShaderVariant


def shader_symbols_from_fx_nodes(
    nodes: Sequence[FxNodeLike],
    *,
    name_map: Mapping[str, str] | None = None,
    project: FxNodeProjector = project_fx_node,
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
) -> tuple[str, ...]:
    return shader_symbols_from_static_nodes(
        project_fx_nodes(nodes, name_map=name_map, project=project),
        lowering_registry=lowering_registry,
    )


def shader_symbols_from_static_nodes(
    nodes: Sequence[StaticNode],
    *,
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
) -> tuple[str, ...]:
    symbols: list[str] = []
    seen: set[str] = set()
    for target, inputs, _outputs in nodes:
        binding = lowering_registry.resolve_target_inputs(target=target, inputs=inputs)
        if binding is None or binding.shader in seen:
            continue
        seen.add(binding.shader)
        symbols.append(binding.shader)
    return tuple(symbols)


def render_dispatch_body_from_fx_nodes(
    nodes: Sequence[FxNodeLike],
    shader_variants: Mapping[str, ShaderVariant],
    *,
    name_map: Mapping[str, str] | None = None,
    project: FxNodeProjector = project_fx_node,
    prefix: str = "tensors",
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
    indent: str = "    ",
) -> str:
    return render_dispatch_body_from_static_nodes(
        project_fx_nodes(nodes, name_map=name_map, project=project),
        shader_variants,
        prefix=prefix,
        lowering_registry=lowering_registry,
        indent=indent,
    )


def render_dispatch_body_from_static_nodes(
    nodes: Sequence[StaticNode],
    shader_variants: Mapping[str, ShaderVariant],
    *,
    prefix: str = "tensors",
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
    indent: str = "    ",
) -> str:
    lines: list[str] = []
    for target, inputs, outputs in nodes:
        binding = lowering_registry.resolve_target_inputs(target=target, inputs=inputs)
        if binding is None:
            lines.append(f"{indent}# UNRESOLVED: {target}")
            continue
        shader = shader_variants.get(binding.shader)
        if shader is None:
            lines.append(f"{indent}# MISSING SHADER: {binding.shader}")
            continue
        lines.append(
            f"{indent}{_render_shader_call(binding.shader, shader, inputs, outputs, prefix)}"
        )
    return "\n".join(lines)


def _render_shader_call(
    shader_symbol: str,
    shader: ShaderVariant,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    prefix: str,
) -> str:
    input_fields = tuple(field for field in shader.contract.fields if field.io_kind is IOKind.INPUT)
    output_fields = tuple(
        field for field in shader.contract.fields if field.io_kind in (IOKind.OUTPUT, IOKind.INOUT)
    )

    args = ["rt"]
    args.extend(
        f"{contract_field.name}={prefix}.{inputs[index]}"
        for index, contract_field in enumerate(input_fields)
        if index < len(inputs)
    )
    args.extend(
        f"{contract_field.name}={prefix}.{outputs[index]}"
        for index, contract_field in enumerate(output_fields)
        if index < len(outputs)
    )
    return f"{shader_symbol}({', '.join(args)})"
