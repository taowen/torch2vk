"""Loop-aware codegen for modules with repeated layers (nn.ModuleList loops).

When torch.export unrolls a for-loop over layers, this module splits the
resulting flat graph into pre-loop / loop-body / post-loop sections and
generates a dispatch function with an actual loop.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import Graph, Node

from torch2vk.export.codegen import (
    SKIP_OPS,
    _AliasOp,
    _DispatchOp,
    _Op,
    _TensorKind,
    _TensorMeta,
    _UnsupportedOp,
    _collect_ops,
    _dedup_variant,
    _find_graph_outputs,
    _prune_dead_ops,
    _resolve_all_variants,
    render_tensor_class,
)
from torch2vk.export.graph import LayerLoopHint, is_alias_op, node_input_names
from torch2vk.export.registry import DEFAULT_REGISTRY, ShaderRegistry
from torch2vk.runtime.shader import ShaderVariant


@dataclass(frozen=True, slots=True)
class _LoopAnalysis:
    pre_ops: list[_Op]
    layer_ops: list[_Op]
    post_ops: list[_Op]
    layer_local_tensors: frozenset[str]
    carry_inputs: list[str]
    carry_output: str | None


def _get_node_layer_index(node: Node, layer_prefix: str) -> int | None:
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return None
    for key in stack:
        key_str = str(key)
        marker = f"L__self__{layer_prefix}."
        if marker in key_str:
            after = key_str.split(marker, 1)[1]
            idx_str = after.split(".")[0].split(",")[0]
            if idx_str.isdigit():
                return int(idx_str)
    return None


def _classify_graph_nodes(graph: Graph, layer_prefix: str) -> dict[str, str]:
    classification: dict[str, str] = {}
    seen_layer = False

    for node in graph.nodes:
        if node.op == "placeholder":
            classification[node.name] = "shared"
            continue
        if node.op != "call_function":
            continue

        layer_idx = _get_node_layer_index(node, layer_prefix)
        if layer_idx is not None:
            seen_layer = True
            classification[node.name] = f"layer.{layer_idx}"
        elif seen_layer:
            classification[node.name] = "post"
        else:
            classification[node.name] = "pre"

    return classification


def _analyze_loop(
    graph: Graph,
    live_ops: list[_Op],
    classification: dict[str, str],
    hint: LayerLoopHint,
) -> _LoopAnalysis:
    pre_ops: list[_Op] = []
    layer0_ops: list[_Op] = []
    post_ops: list[_Op] = []

    for op in live_ops:
        name = _op_name(op)
        section = classification.get(name, "pre")
        if section == "pre":
            pre_ops.append(op)
        elif section == "layer.0":
            layer0_ops.append(op)
        elif section == "post":
            post_ops.append(op)

    # Identify layer-local tensors: outputs of layer.0 ops + their layer.0 inputs
    layer_local: set[str] = set()
    for op in layer0_ops:
        if isinstance(op, _DispatchOp):
            layer_local.add(op.name)
            for v in op.bindings.values():
                t = v.removeprefix("tensors.")
                if classification.get(t, "").startswith("layer."):
                    layer_local.add(t)
        elif isinstance(op, _AliasOp):
            layer_local.add(op.dst)
            if classification.get(op.src, "").startswith("layer."):
                layer_local.add(op.src)

    # Detect carry: layer.0 inputs that come from pre-loop intermediates (not shared/params)
    pre_intermediates = {_op_name(op) for op in pre_ops}
    carry_inputs: list[str] = []
    for op in layer0_ops:
        if isinstance(op, _DispatchOp):
            for v in op.bindings.values():
                t = v.removeprefix("tensors.")
                if t in pre_intermediates and t not in carry_inputs:
                    carry_inputs.append(t)
        elif isinstance(op, _AliasOp):
            if op.src in pre_intermediates and op.src not in carry_inputs:
                carry_inputs.append(op.src)

    # Detect carry output: layer.0 output that layer.1 uses in the same role
    # (the tensor from layer.0 that replaces the carry_input for layer.1)
    carry_output: str | None = None
    if carry_inputs and hint.num_layers > 1:
        layer1_inputs_from_layer0: set[str] = set()
        for op in live_ops:
            name = _op_name(op)
            if classification.get(name) != "layer.1":
                continue
            if isinstance(op, _DispatchOp):
                for v in op.bindings.values():
                    t = v.removeprefix("tensors.")
                    if classification.get(t) == "layer.0":
                        layer1_inputs_from_layer0.add(t)
            elif isinstance(op, _AliasOp):
                if classification.get(op.src) == "layer.0":
                    layer1_inputs_from_layer0.add(op.src)

        if layer1_inputs_from_layer0:
            carry_output = next(iter(layer1_inputs_from_layer0))

    return _LoopAnalysis(
        pre_ops=pre_ops,
        layer_ops=layer0_ops,
        post_ops=post_ops,
        layer_local_tensors=frozenset(layer_local),
        carry_inputs=carry_inputs,
        carry_output=carry_output,
    )


def _op_name(op: _Op) -> str:
    if isinstance(op, _DispatchOp):
        return op.name
    elif isinstance(op, _AliasOp):
        return op.dst
    elif isinstance(op, _UnsupportedOp):
        return op.name
    return ""


def _layer_param_names(sig, weight_prefix: str) -> frozenset[str]:
    """Identify placeholder names that are layer.0 parameters."""
    names: set[str] = set()
    for spec in sig.input_specs:
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
            key = f"{weight_prefix}{spec.target}"
            if re.search(r"\.layers\.0\.", key):
                names.add(spec.arg.name)
    return frozenset(names)


def generate_looped_dispatch_function_source(
    prog: ExportedProgram,
    *,
    parent_class_name: str,
    layer_class_name: str,
    function_name: str,
    weight_prefix: str = "",
    hint: LayerLoopHint,
    registry: ShaderRegistry = DEFAULT_REGISTRY,
) -> tuple[str, dict[str, str], dict[str, ShaderVariant]]:
    """Generate dispatch function with a loop over layers.

    Returns (function_source, {shader_name: CONST_NAME}, {shader_name: ShaderVariant}).
    """
    graph = prog.graph_module.graph
    node_variants = _resolve_all_variants(graph, registry)
    all_ops = _collect_ops(graph, node_variants)
    output_names = _find_graph_outputs(graph)
    live_ops = _prune_dead_ops(all_ops, output_names)

    classification = _classify_graph_nodes(graph, hint.layer_prefix)
    analysis = _analyze_loop(graph, live_ops, classification, hint)

    # Layer params are also layer-local (they're placeholders, not in nn_module_stack)
    layer_params = _layer_param_names(prog.graph_signature, weight_prefix)
    layer_local = analysis.layer_local_tensors | layer_params

    shader_imports: dict[str, str] = {}
    carry_set = frozenset(analysis.carry_inputs)

    def _tensor_ref(tensor_name: str, in_loop: bool) -> str:
        if in_loop and tensor_name in carry_set:
            return "carry"
        if in_loop and tensor_name in layer_local:
            return f"layer_t.{tensor_name}"
        return f"tensors.{tensor_name}"

    def _emit_op(op: _Op, in_loop: bool, indent: str) -> str:
        if isinstance(op, _AliasOp):
            src = _tensor_ref(op.src, in_loop)
            dst = _tensor_ref(op.dst, in_loop)
            return f"{indent}_alias(rt, {src}, {dst})"
        elif isinstance(op, _DispatchOp):
            const_name = op.variant.name.upper()
            shader_imports[op.variant.name] = const_name
            bindings: list[str] = []
            for k, v in op.bindings.items():
                t = v.removeprefix("tensors.")
                bindings.append(f"{k}={_tensor_ref(t, in_loop)}")
            return f"{indent}{const_name}(rt, {', '.join(bindings)})"
        elif isinstance(op, _UnsupportedOp):
            return f"{indent}raise RuntimeError({op.message!r})"
        return ""

    lines: list[str] = []
    lines.append(f"def {function_name}(rt: RuntimeSession, tensors: {parent_class_name}) -> None:")

    # Pre-loop
    for op in analysis.pre_ops:
        lines.append(_emit_op(op, False, "    "))

    # Initialize carry
    if analysis.carry_inputs:
        carry_init = analysis.carry_inputs[0]
        lines.append(f"    carry = tensors.{carry_init}")

    # Loop
    lines.append(f"    for layer_t in tensors.layers:")
    for op in analysis.layer_ops:
        lines.append(_emit_op(op, True, "        "))
    if analysis.carry_output:
        lines.append(f"        carry = layer_t.{analysis.carry_output}")

    # Post-loop: rewrite references to last layer's output → carry
    # Find what post ops reference from layer.N-1 and replace with carry
    last_layer_prefix = f"layer.{hint.num_layers - 1}"
    post_carry_targets: set[str] = set()
    for op in analysis.post_ops:
        if isinstance(op, _DispatchOp):
            for v in op.bindings.values():
                t = v.removeprefix("tensors.")
                if classification.get(t, "").startswith("layer."):
                    post_carry_targets.add(t)
        elif isinstance(op, _AliasOp):
            if classification.get(op.src, "").startswith("layer."):
                post_carry_targets.add(op.src)

    for op in analysis.post_ops:
        if isinstance(op, _AliasOp):
            src = "carry" if op.src in post_carry_targets else f"tensors.{op.src}"
            dst = f"tensors.{op.dst}"
            lines.append(f"    _alias(rt, {src}, {dst})")
        elif isinstance(op, _DispatchOp):
            const_name = op.variant.name.upper()
            shader_imports[op.variant.name] = const_name
            bindings: list[str] = []
            for k, v in op.bindings.items():
                t = v.removeprefix("tensors.")
                if t in post_carry_targets:
                    bindings.append(f"{k}=carry")
                else:
                    bindings.append(f"{k}=tensors.{t}")
            lines.append(f"    {const_name}(rt, {', '.join(bindings)})")
        elif isinstance(op, _UnsupportedOp):
            lines.append(f"    raise RuntimeError({op.message!r})")

    used_variants: dict[str, ShaderVariant] = {}
    for op in analysis.pre_ops + analysis.layer_ops + analysis.post_ops:
        if isinstance(op, _DispatchOp):
            used_variants.setdefault(op.variant.name, op.variant)

    return "\n".join(lines), shader_imports, used_variants


def generate_looped_tensor_class_sources(
    prog: ExportedProgram,
    *,
    parent_class_name: str,
    layer_class_name: str,
    parent_function_name: str,
    layer_function_name: str,
    weight_prefix: str = "",
    hint: LayerLoopHint,
    extra_lines_fn: Callable[[str], tuple[str, ...]] | None = None,
    registry: ShaderRegistry = DEFAULT_REGISTRY,
) -> tuple[str, str]:
    """Generate parent + layer tensor classes.

    Returns (parent_class_source, layer_class_source).
    """
    graph = prog.graph_module.graph
    sig = prog.graph_signature
    node_variants = _resolve_all_variants(graph, registry)

    # Collect tensor metadata
    tensors: dict[str, _TensorMeta] = {}
    user_inputs: list[str] = []
    param_map: dict[str, str] = {}

    for spec in sig.input_specs:
        for node in graph.nodes:
            if node.name == spec.arg.name:
                tm = node.meta.get("tensor_meta")
                if tm:
                    shape = tuple(int(d) for d in tm.shape)
                    dtype = str(tm.dtype).removeprefix("torch.")
                    is_param = spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
                    tensors[spec.arg.name] = _TensorMeta(
                        shape=shape, dtype=dtype,
                        kind=_TensorKind.PARAMETER if is_param else _TensorKind.USER_INPUT,
                    )
                    if is_param:
                        param_map[spec.arg.name] = f"{weight_prefix}{spec.target}"
                    else:
                        user_inputs.append(spec.arg.name)
                break

    for node in graph.nodes:
        if node.op == "call_function" and node.name not in tensors:
            if str(node.target) in SKIP_OPS:
                continue
            tm = node.meta.get("tensor_meta")
            if tm:
                shape = tuple(int(d) for d in tm.shape)
                dtype = str(tm.dtype).removeprefix("torch.")
                tensors[node.name] = _TensorMeta(shape=shape, dtype=dtype, kind=_TensorKind.INTERMEDIATE)

    # Prune dead tensors using full graph
    all_ops = _collect_ops(graph, node_variants)
    output_names = _find_graph_outputs(graph)
    live_ops = _prune_dead_ops(all_ops, output_names)
    live_tensors = set(output_names)
    for op in live_ops:
        if isinstance(op, _DispatchOp):
            for v in op.bindings.values():
                live_tensors.add(v.removeprefix("tensors."))
        elif isinstance(op, _AliasOp):
            live_tensors.add(op.src)
            live_tensors.add(op.dst)
    live_tensors.update(name for name in user_inputs if name in live_tensors)
    live_tensors.update(param_map.keys() & live_tensors)
    tensors = {k: v for k, v in tensors.items() if k in live_tensors}
    user_inputs = [n for n in user_inputs if n in live_tensors]
    param_map = {k: v for k, v in param_map.items() if k in live_tensors}

    # Analyze loop structure
    classification = _classify_graph_nodes(graph, hint.layer_prefix)
    analysis = _analyze_loop(graph, live_ops, classification, hint)

    # Classify parameters: those with .layers.0. go to layer class
    layer_param_map: dict[str, str] = {}
    parent_param_map: dict[str, str] = {}
    for name, safetensors_key in param_map.items():
        if re.search(r"\.layers\.0\.", safetensors_key):
            layer_param_map[name] = safetensors_key
        elif re.search(r"\.layers\.\d+\.", safetensors_key):
            pass  # skip layers.1+ params (duplicates)
        else:
            parent_param_map[name] = safetensors_key

    # Split tensors: layer-local intermediates + layer params → layer class
    layer_tensors: dict[str, _TensorMeta] = {}
    parent_tensors: dict[str, _TensorMeta] = {}

    for name, meta in tensors.items():
        if name in layer_param_map:
            layer_tensors[name] = meta
        elif name in parent_param_map:
            parent_tensors[name] = meta
        elif meta.kind == _TensorKind.PARAMETER:
            pass  # layers.1+ params — handled by layer_param_map pattern
        elif meta.kind == _TensorKind.USER_INPUT:
            parent_tensors[name] = meta
        elif name in analysis.layer_local_tensors:
            layer_tensors[name] = meta
        elif classification.get(name, "").startswith("layer.0"):
            layer_tensors[name] = meta
        elif classification.get(name, "").startswith("layer."):
            pass  # skip layer.1+ intermediates
        else:
            parent_tensors[name] = meta

    # Find module output
    output_name = None
    for node in graph.nodes:
        if node.op == "output":
            for arg in _flatten_args(node.args):
                if isinstance(arg, Node) and arg.name in parent_tensors:
                    output_name = arg.name
                    break
            break

    # Generate layer tensor class
    layer_src = _render_layer_class(
        tensors=layer_tensors,
        param_map=layer_param_map,
        class_name=layer_class_name,
        function_name=layer_function_name,
        weight_prefix=weight_prefix,
        extra_lines_fn=extra_lines_fn,
    )

    # Generate parent tensor class
    parent_src = _render_parent_class(
        tensors=parent_tensors,
        param_map=parent_param_map,
        user_inputs=user_inputs,
        class_name=parent_class_name,
        function_name=parent_function_name,
        layer_class_name=layer_class_name,
        layer_function_name=layer_function_name,
        num_layers=hint.num_layers,
        output_name=output_name,
        extra_lines_fn=extra_lines_fn,
    )

    return parent_src, layer_src


def _flatten_args(args) -> list:
    result = []
    if isinstance(args, (list, tuple)):
        for item in args:
            result.extend(_flatten_args(item))
    else:
        result.append(args)
    return result


def _render_layer_class(
    *,
    tensors: dict[str, _TensorMeta],
    param_map: dict[str, str],
    class_name: str,
    function_name: str,
    weight_prefix: str,
    extra_lines_fn: Callable[[str], tuple[str, ...]] | None,
) -> str:
    sig_str = (
        f"def {function_name}(prefix: str, layer_idx: int, "
        f"*, bindings: Mapping[str, LogicalTensor] | None = None, "
        f"request_state_outputs: Collection[str] = frozenset()) -> {class_name}:"
    )

    tensor_entries = []
    for name, meta in tensors.items():
        kind = meta.kind
        dtype = "bfloat16" if kind == _TensorKind.PARAMETER else (
            meta.dtype if meta.dtype in ("int64", "int32") else "float32"
        )
        if kind == _TensorKind.PARAMETER:
            safetensors_key = param_map[name]
            name_template = re.sub(r"\.layers\.0\.", ".layers.{layer_idx}.", safetensors_key)
            tensor_entries.append({
                "name": name,
                "name_source": repr(name),
                "name_expr": f'f"{name_template}"',
                "dtype_source": repr(dtype),
                "shape_source": repr(meta.shape),
                "role": "TensorRole.WEIGHT",
                "memory": "MemoryClass.MODEL_WEIGHT",
                "lifetime": "TensorLifetime.MODEL",
                "extra_lines": extra_lines_fn(name) if extra_lines_fn else (),
            })
        else:
            tensor_entries.append({
                "name": name,
                "name_source": repr(name),
                "name_expr": f'f"{{prefix}}.layers.{{layer_idx}}.{name}"',
                "dtype_source": repr(dtype),
                "shape_source": repr(meta.shape),
                "role": "TensorRole.ACTIVATION",
                "memory": "MemoryClass.FRAME_WORKSPACE",
                "lifetime": "TensorLifetime.FRAME",
                "extra_lines": extra_lines_fn(name) if extra_lines_fn else (),
            })

    output_name = next(
        (n for n in reversed(tensors) if tensors[n].kind == _TensorKind.INTERMEDIATE),
        next(iter(tensors), "unknown"),
    )
    output_const = function_name.removeprefix("create_").upper() + "_OUTPUT"

    return render_tensor_class(
        class_name=class_name,
        fields=tuple(tensors.keys()),
        output_const=output_const,
        output_name_source=repr(output_name),
        signature=sig_str,
        tensor_names_source=repr(tuple(tensors.keys())),
        tensors=tensor_entries,
    )


def _render_parent_class(
    *,
    tensors: dict[str, _TensorMeta],
    param_map: dict[str, str],
    user_inputs: list[str],
    class_name: str,
    function_name: str,
    layer_class_name: str,
    layer_function_name: str,
    num_layers: int,
    output_name: str | None,
    extra_lines_fn: Callable[[str], tuple[str, ...]] | None,
) -> str:
    sig_str = (
        f"def {function_name}(prefix: str, "
        f"*, bindings: Mapping[str, LogicalTensor] | None = None, "
        f"request_state_outputs: Collection[str] = frozenset()) -> {class_name}:"
    )

    tensor_entries = []
    for name, meta in tensors.items():
        kind = meta.kind
        dtype = "bfloat16" if kind == _TensorKind.PARAMETER else (
            meta.dtype if meta.dtype in ("int64", "int32") else "float32"
        )
        if kind == _TensorKind.PARAMETER:
            tensor_entries.append({
                "name": name,
                "name_source": repr(name),
                "name_expr": f'"{param_map[name]}"',
                "dtype_source": repr(dtype),
                "shape_source": repr(meta.shape),
                "role": "TensorRole.WEIGHT",
                "memory": "MemoryClass.MODEL_WEIGHT",
                "lifetime": "TensorLifetime.MODEL",
                "extra_lines": extra_lines_fn(name) if extra_lines_fn else (),
            })
        elif kind == _TensorKind.USER_INPUT:
            tensor_entries.append({
                "name": name,
                "name_source": repr(name),
                "name_expr": f'f"{{prefix}}.{name}"',
                "dtype_source": repr(dtype),
                "shape_source": repr(meta.shape),
                "role": "TensorRole.INPUT",
                "memory": "MemoryClass.HOST_INPUT",
                "lifetime": "TensorLifetime.FRAME",
                "extra_lines": extra_lines_fn(name) if extra_lines_fn else (),
            })
        else:
            tensor_entries.append({
                "name": name,
                "name_source": repr(name),
                "name_expr": f'f"{{prefix}}.{name}"',
                "dtype_source": repr(dtype),
                "shape_source": repr(meta.shape),
                "role": "TensorRole.ACTIVATION",
                "memory": "MemoryClass.FRAME_WORKSPACE",
                "lifetime": "TensorLifetime.FRAME",
                "extra_lines": extra_lines_fn(name) if extra_lines_fn else (),
            })

    if output_name is None:
        output_name = next(
            (n for n in reversed(tensors) if tensors[n].kind == _TensorKind.INTERMEDIATE),
            next(iter(tensors), "unknown"),
        )

    output_const = function_name.removeprefix("create_").upper() + "_OUTPUT"

    # Build manually (includes layers field)
    lines: list[str] = []
    lines.append("@dataclass(frozen=True, slots=True)")
    lines.append(f"class {class_name}:")
    for name in tensors:
        lines.append(f"    {name}: LogicalTensor")
    lines.append(f"    layers: list[{layer_class_name}]")
    lines.append("")
    lines.append("")
    lines.append(f"{output_const}: str = {output_name!r}")
    lines.append("")
    lines.append("")
    lines.append(sig_str)
    lines.append(f"    _validate_bindings(bindings, frozenset({tuple(tensors.keys())!r}))")
    lines.append(f"    _validate_request_state_outputs(request_state_outputs, frozenset(({output_name!r},)))")
    lines.append(f"    return {class_name}(")
    for entry in tensor_entries:
        lines.append(f"        {entry['name']}=_bind_tensor(")
        lines.append(f"            bindings,")
        lines.append(f"            {entry['name_source']},")
        lines.append(f"            _declare_tensor(")
        lines.append(f"            name={entry['name_expr']},")
        lines.append(f"            spec=TensorSpec(dtype={entry['dtype_source']}, shape={entry['shape_source']}),")
        lines.append(f"            role={entry['role']},")
        lines.append(f"            memory={entry['memory']},")
        lines.append(f"            lifetime={entry['lifetime']},")
        lines.append(f"            request_state={entry['name_source']} in request_state_outputs,")
        for extra_line in entry["extra_lines"]:
            lines.append(f"            {extra_line}")
        lines.append("            ),")
        lines.append("        ),")
    lines.append(f"        layers=[{layer_function_name}(prefix, layer_idx=i) for i in range({num_layers})],")
    lines.append("    )")
    return "\n".join(lines)
