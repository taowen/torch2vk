"""Frame module generation from static node declarations and shader contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from torch2vk.exportv2.lowering import DEFAULT_LOWERING_REGISTRY, OpLoweringRegistry
from torch2vk.exportv2.writer import RenderedFile, format_python_source
from torch2vk.exportv2.dispatch_codegen import (
    render_dispatch_body_from_static_nodes,
    shader_symbols_from_static_nodes,
)
from torch2vk.exportv2.fx import StaticNode
from torch2vk.runtime.shader import IOKind, ShaderVariant

TensorRef = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class FrameSpec:
    function_name: str
    frame_name: str
    tensors_class: str
    tensors_import: str
    shader_import: str
    nodes: tuple[StaticNode, ...]
    return_expr: str | None = None
    layer_loop_target: str | None = None
    layer_nodes: tuple[StaticNode, ...] | None = None
    layer_state_field: str | None = None
    layer_state_init: str | None = None
    layer_external_fields: dict[str, str] = field(default_factory=dict)
    frame_preamble: str | None = None
    extra_imports: tuple[str, ...] = ()
    extra_params: str | None = None


def render_frame_module(
    spec: FrameSpec,
    shader_variants: Mapping[str, ShaderVariant],
    *,
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
) -> RenderedFile:
    lines: list[str] = []
    shader_symbols: set[str] = set()

    _collect_shader_symbols(spec.nodes, shader_variants, lowering_registry, shader_symbols)
    if spec.layer_nodes:
        _collect_shader_symbols(
            spec.layer_nodes, shader_variants, lowering_registry, shader_symbols
        )

    _emit_header(lines, spec, sorted(shader_symbols))
    lines.extend(("", ""))
    _emit_function(lines, spec, shader_variants, lowering_registry)

    source = "\n".join(lines) + "\n"
    formatted = format_python_source(source, filename=f"{spec.frame_name}.py")
    parts = spec.function_name.removeprefix("run_generated_qwen3_asr_").removeprefix(
        "run_omnivoice_"
    )
    return RenderedFile(relative_path=Path(f"{parts}.py"), content=formatted)


def _collect_shader_symbols(
    nodes: tuple[StaticNode, ...],
    shader_variants: Mapping[str, ShaderVariant],
    lowering_registry: OpLoweringRegistry,
    out: set[str],
) -> None:
    out.update(
        symbol
        for symbol in shader_symbols_from_static_nodes(
            nodes,
            lowering_registry=lowering_registry,
        )
        if symbol in shader_variants
    )


def _emit_header(lines: list[str], spec: FrameSpec, shader_symbols: list[str]) -> None:
    lines.append(f'"""Generated {spec.frame_name} frame scaffold."""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    if spec.return_expr:
        lines.append("from torch2vk.runtime.logical import LogicalTensor")
    lines.append("from torch2vk.runtime.session import RuntimeSession")
    if shader_symbols:
        lines.append(f"from {spec.shader_import} import (")
        lines.extend(f"    {symbol}," for symbol in shader_symbols)
        lines.append(")")
    if spec.tensors_import and spec.tensors_class:
        lines.append(f"from {spec.tensors_import} import {spec.tensors_class}")
    lines.extend(spec.extra_imports)


def _emit_function(
    lines: list[str],
    spec: FrameSpec,
    shader_variants: Mapping[str, ShaderVariant],
    lowering_registry: OpLoweringRegistry,
) -> None:
    lines.append(f"def {spec.function_name}(")
    lines.append("    rt: RuntimeSession,")
    lines.append(f"    tensors: {spec.tensors_class},")
    if spec.extra_params:
        lines.append(spec.extra_params)
    return_type = "LogicalTensor" if spec.return_expr else "None"
    lines.append(f") -> {return_type}:")

    if spec.frame_preamble:
        for preamble_line in spec.frame_preamble.splitlines():
            lines.append(f"    {preamble_line}" if preamble_line.strip() else "")

    lines.append(f"    with rt.frame({spec.frame_name!r}):")

    state_field = spec.layer_state_field
    after_layer_loop = False
    for target, inputs, outputs in spec.nodes:
        if spec.layer_loop_target and target == spec.layer_loop_target:
            _emit_layer_loop(lines, spec, shader_variants, lowering_registry)
            after_layer_loop = True
            continue

        node = (target, inputs, outputs)
        if after_layer_loop and state_field:
            call = _render_post_loop_node(node, shader_variants, lowering_registry, state_field)
            lines.append(f"        {call}")
        else:
            body = render_dispatch_body_from_static_nodes(
                (node,),
                shader_variants,
                lowering_registry=lowering_registry,
                indent="        ",
            )
            lines.append(body)

    if spec.return_expr:
        lines.append(f"    return {spec.return_expr}")


def _emit_layer_loop(
    lines: list[str],
    spec: FrameSpec,
    shader_variants: Mapping[str, ShaderVariant],
    lowering_registry: OpLoweringRegistry,
) -> None:
    if not spec.layer_nodes:
        return
    state = spec.layer_state_field or "hidden"
    init = spec.layer_state_init or f"tensors.{state}"
    external = spec.layer_external_fields or {}

    lines.append(f"        {state} = {init}")
    lines.append("        for layer in tensors.layers:")

    for node in spec.layer_nodes:
        call = _render_layer_node(node, shader_variants, lowering_registry, state, external)
        lines.append(f"            {call}")

    lines.append(f"            {state} = layer.{spec.layer_nodes[-1][2][0]}")


def _render_post_loop_node(
    node: StaticNode,
    shader_variants: Mapping[str, ShaderVariant],
    lowering_registry: OpLoweringRegistry,
    state_field: str,
) -> str:
    target, inputs, outputs = node
    binding = lowering_registry.resolve_target_inputs(target=target, inputs=inputs)
    if binding is None:
        return f"# UNRESOLVED: {target}"
    shader = shader_variants.get(binding.shader)
    if shader is None:
        return f"# MISSING SHADER: {binding.shader}"
    return _render_shader_call(
        binding.shader,
        shader,
        inputs,
        outputs,
        lambda name: state_field if name == state_field else f"tensors.{name}",
    )


def _render_layer_node(
    node: StaticNode,
    shader_variants: Mapping[str, ShaderVariant],
    lowering_registry: OpLoweringRegistry,
    state_field: str,
    external_fields: dict[str, str],
) -> str:
    target, inputs, outputs = node
    binding = lowering_registry.resolve_target_inputs(target=target, inputs=inputs)
    if binding is None:
        return f"# UNRESOLVED: {target}"
    shader = shader_variants.get(binding.shader)
    if shader is None:
        return f"# MISSING SHADER: {binding.shader}"
    return _render_shader_call(
        binding.shader,
        shader,
        inputs,
        outputs,
        lambda name: _layer_ref(name, state_field, external_fields),
    )


def _render_shader_call(
    shader_symbol: str,
    shader: ShaderVariant,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    ref: TensorRef,
) -> str:
    input_fields = tuple(field for field in shader.contract.fields if field.io_kind is IOKind.INPUT)
    output_fields = tuple(
        field for field in shader.contract.fields if field.io_kind in (IOKind.OUTPUT, IOKind.INOUT)
    )
    args = ["rt"]
    for index, contract_field in enumerate(input_fields):
        if index < len(inputs):
            args.append(f"{contract_field.name}={ref(inputs[index])}")
    for index, contract_field in enumerate(output_fields):
        if index < len(outputs):
            args.append(f"{contract_field.name}={ref(outputs[index])}")
    return f"{shader_symbol}({', '.join(args)})"


def _layer_ref(tensor_name: str, state_field: str, external_fields: dict[str, str]) -> str:
    if tensor_name == state_field:
        return state_field
    if tensor_name in external_fields:
        return external_fields[tensor_name]
    return f"layer.{tensor_name}"
