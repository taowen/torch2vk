"""Generate frame dispatch modules from op declarations and shader contracts.

Given a sequence of ops and a lowering registry, produces Python source with
static shader call sequences — the same style as hand-written frame code.
Resolution happens at generation time; the output has no runtime dispatch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from torch2vk.export.lowering import DEFAULT_LOWERING_REGISTRY, OpLoweringRegistry
from torch2vk.export.writer import RenderedFile, format_python_source
from torch2vk.runtime.shader import IOKind, ShaderVariant


@dataclass(frozen=True, slots=True)
class FrameSpec:
    """Specification for generating a frame dispatch module."""

    function_name: str
    frame_name: str
    tensors_class: str
    tensors_import: str
    ops: tuple[dict, ...]
    return_expr: str | None = None
    # Layer loop
    layer_loop_target: str | None = None
    layer_ops: tuple[dict, ...] | None = None
    layer_state_field: str | None = None
    layer_state_init: str | None = None
    layer_external_fields: dict[str, str] = field(default_factory=dict)
    # Frame extras
    frame_preamble: str | None = None
    extra_imports: tuple[str, ...] = ()
    extra_params: str | None = None


def render_frame_module(
    spec: FrameSpec,
    shader_variants: Mapping[str, ShaderVariant],
    *,
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
) -> RenderedFile:
    """Generate a complete frame .py file with static shader call sequences."""
    lines: list[str] = []
    shader_symbols: set[str] = set()

    _collect_shader_symbols(spec.ops, shader_variants, lowering_registry, shader_symbols)
    if spec.layer_ops:
        _collect_shader_symbols(spec.layer_ops, shader_variants, lowering_registry, shader_symbols)

    _emit_header(lines, spec, sorted(shader_symbols))
    lines.append("")
    lines.append("")
    _emit_function(lines, spec, shader_variants, lowering_registry)

    source = "\n".join(lines) + "\n"
    formatted = format_python_source(source, filename=f"{spec.frame_name}.py")
    fn_suffix = spec.function_name.split("_", 4)[-1] if "_" in spec.function_name else spec.function_name
    relative_path = fn_suffix.rsplit("_", 1)[-1] + ".py" if "_" in fn_suffix else f"{fn_suffix}.py"
    # Use a simpler naming: strip the common prefix
    parts = spec.function_name.removeprefix("run_generated_qwen3_asr_").removeprefix("run_omnivoice_")
    relative_path = f"{parts}.py"
    return RenderedFile(relative_path=Path(relative_path), content=formatted)


def render_dispatch_body(
    ops: Sequence[dict],
    shader_variants: Mapping[str, ShaderVariant],
    *,
    prefix: str = "tensors",
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
    indent: str = "    ",
) -> str:
    """Generate static shader call sequence as source lines."""
    lines: list[str] = []
    for op in ops:
        target = op["target"]
        inputs = tuple(op["inputs"])
        outputs = tuple(op["outputs"])

        binding = lowering_registry.resolve_target_inputs(target=target, inputs=inputs)
        if binding is None:
            lines.append(f"{indent}# UNRESOLVED: {target}")
            continue

        shader = shader_variants.get(binding.shader)
        if shader is None:
            lines.append(f"{indent}# MISSING SHADER: {binding.shader}")
            continue

        call = _render_shader_call(binding.shader, shader, inputs, outputs, prefix)
        lines.append(f"{indent}{call}")

    return "\n".join(lines)


def _collect_shader_symbols(
    ops: Sequence[dict],
    shader_variants: Mapping[str, ShaderVariant],
    lowering_registry: OpLoweringRegistry,
    out: set[str],
) -> None:
    for op in ops:
        target = op["target"]
        inputs = tuple(op["inputs"])
        binding = lowering_registry.resolve_target_inputs(target=target, inputs=inputs)
        if binding is not None and binding.shader in shader_variants:
            out.add(binding.shader)


def _emit_header(lines: list[str], spec: FrameSpec, shader_symbols: list[str]) -> None:
    lines.append(f'"""Generated {spec.frame_name} frame scaffold."""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from torch2vk.runtime.session import RuntimeSession")
    if shader_symbols:
        lines.append("from torch2vk.export.shaders import (")
        for sym in shader_symbols:
            lines.append(f"    {sym},")
        lines.append(")")
    if spec.tensors_import and spec.tensors_class:
        lines.append(f"from {spec.tensors_import} import {spec.tensors_class}")
    for imp in spec.extra_imports:
        lines.append(imp)


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
    lines.append(") -> None:")

    if spec.frame_preamble:
        for preamble_line in spec.frame_preamble.splitlines():
            lines.append(f"    {preamble_line}" if preamble_line.strip() else "")

    lines.append(f'    with rt.frame({spec.frame_name!r}):')

    state_field = spec.layer_state_field
    after_layer_loop = False

    for op in spec.ops:
        target = op["target"]
        inputs = tuple(op["inputs"])
        outputs = tuple(op["outputs"])

        if spec.layer_loop_target and target == spec.layer_loop_target:
            _emit_layer_loop(lines, spec, shader_variants, lowering_registry)
            after_layer_loop = True
            continue

        binding = lowering_registry.resolve_target_inputs(target=target, inputs=inputs)
        if binding is None:
            lines.append(f"        # UNRESOLVED: {target}")
            continue
        shader = shader_variants.get(binding.shader)
        if shader is None:
            lines.append(f"        # MISSING SHADER: {binding.shader}")
            continue

        if after_layer_loop and state_field:
            call = _render_post_loop_shader_call(
                binding.shader, shader, inputs, outputs, "tensors", state_field
            )
        else:
            call = _render_shader_call(binding.shader, shader, inputs, outputs, "tensors")
        lines.append(f"        {call}")

    if spec.return_expr:
        lines.append(f"    return {spec.return_expr}")


def _emit_layer_loop(
    lines: list[str],
    spec: FrameSpec,
    shader_variants: Mapping[str, ShaderVariant],
    lowering_registry: OpLoweringRegistry,
) -> None:
    if not spec.layer_ops:
        return
    state = spec.layer_state_field or "hidden"
    init = spec.layer_state_init or f"tensors.{state}"
    external = spec.layer_external_fields or {}

    lines.append(f"        {state} = {init}")
    lines.append("        for layer in tensors.layers:")

    for op in spec.layer_ops:
        target = op["target"]
        inputs = tuple(op["inputs"])
        outputs = tuple(op["outputs"])

        binding = lowering_registry.resolve_target_inputs(target=target, inputs=inputs)
        if binding is None:
            lines.append(f"            # UNRESOLVED: {target}")
            continue
        shader = shader_variants.get(binding.shader)
        if shader is None:
            lines.append(f"            # MISSING SHADER: {binding.shader}")
            continue

        call = _render_layer_shader_call(
            binding.shader, shader, inputs, outputs,
            state_field=state,
            external_fields=external,
        )
        lines.append(f"            {call}")

    last_output = spec.layer_ops[-1]["outputs"][0] if spec.layer_ops else "output"
    lines.append(f"            {state} = layer.{last_output}")


def _render_shader_call(
    shader_symbol: str,
    shader: ShaderVariant,
    op_inputs: tuple[str, ...],
    op_outputs: tuple[str, ...],
    prefix: str,
) -> str:
    """Render a shader call with prefix.field access."""
    input_fields = [f for f in shader.contract.fields if f.io_kind is IOKind.INPUT]
    output_fields = [
        f for f in shader.contract.fields if f.io_kind in (IOKind.OUTPUT, IOKind.INOUT)
    ]

    args: list[str] = ["rt"]
    for i, contract_field in enumerate(input_fields):
        if i < len(op_inputs):
            args.append(f"{contract_field.name}={prefix}.{op_inputs[i]}")
    for j, contract_field in enumerate(output_fields):
        if j < len(op_outputs):
            args.append(f"{contract_field.name}={prefix}.{op_outputs[j]}")

    return f"{shader_symbol}({', '.join(args)})"


def _render_post_loop_shader_call(
    shader_symbol: str,
    shader: ShaderVariant,
    op_inputs: tuple[str, ...],
    op_outputs: tuple[str, ...],
    prefix: str,
    state_field: str,
) -> str:
    """Render a shader call after the layer loop — state_field references local var."""
    input_fields = [f for f in shader.contract.fields if f.io_kind is IOKind.INPUT]
    output_fields = [
        f for f in shader.contract.fields if f.io_kind in (IOKind.OUTPUT, IOKind.INOUT)
    ]

    args: list[str] = ["rt"]
    for i, contract_field in enumerate(input_fields):
        if i < len(op_inputs):
            name = op_inputs[i]
            ref = state_field if name == state_field else f"{prefix}.{name}"
            args.append(f"{contract_field.name}={ref}")
    for j, contract_field in enumerate(output_fields):
        if j < len(op_outputs):
            args.append(f"{contract_field.name}={prefix}.{op_outputs[j]}")

    return f"{shader_symbol}({', '.join(args)})"


def _render_layer_shader_call(
    shader_symbol: str,
    shader: ShaderVariant,
    op_inputs: tuple[str, ...],
    op_outputs: tuple[str, ...],
    *,
    state_field: str,
    external_fields: dict[str, str],
) -> str:
    """Render a shader call inside a layer loop.

    - state_field (e.g. "hidden") → local variable `hidden`
    - external_fields (e.g. {"rope_cos": "tensors.rope_cos"}) → parent access
    - everything else → `layer.field`
    """
    input_fields = [f for f in shader.contract.fields if f.io_kind is IOKind.INPUT]
    output_fields = [
        f for f in shader.contract.fields if f.io_kind in (IOKind.OUTPUT, IOKind.INOUT)
    ]

    args: list[str] = ["rt"]
    for i, contract_field in enumerate(input_fields):
        if i < len(op_inputs):
            tensor_name = op_inputs[i]
            args.append(f"{contract_field.name}={_layer_ref(tensor_name, state_field, external_fields)}")
    for j, contract_field in enumerate(output_fields):
        if j < len(op_outputs):
            tensor_name = op_outputs[j]
            args.append(f"{contract_field.name}={_layer_ref(tensor_name, state_field, external_fields)}")

    return f"{shader_symbol}({', '.join(args)})"


def _layer_ref(tensor_name: str, state_field: str, external_fields: dict[str, str]) -> str:
    """Determine the correct reference for a tensor name inside a layer loop."""
    if tensor_name == state_field:
        return state_field
    if tensor_name in external_fields:
        return external_fields[tensor_name]
    return f"layer.{tensor_name}"
