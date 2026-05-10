"""Loop-aware codegen for modules with repeated layers (nn.ModuleList loops).

When torch.export unrolls a for-loop over layers, this module splits the
resulting flat graph into pre-loop / loop-body / post-loop sections and
generates a dispatch function with an actual loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import Graph, Node

from torch2vk.export.dispatch_codegen import (
    _AliasOp,
    _DispatchOp,
    _Op,
    _UnsupportedOp,
    _collect_ops,
    _find_graph_outputs,
    _prune_dead_ops,
    _resolve_all_variants,
)
from torch2vk.export.graph import SKIP_OPS, LayerLoopHint
from torch2vk.export.tensor_codegen import (
    TensorClassContext,
    _TensorKind,
    _TensorMeta,
    _tensor_factory_signature,
    render_tensor_class,
)
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
    tensor_metas: dict[str, _TensorMeta],
    parameter_names: frozenset[str],
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

    # Detect carry: the layer.0 external input that layer.1 replaces with layer.0 output.
    layer0_external_inputs: list[str] = []
    for op in layer0_ops:
        if isinstance(op, _DispatchOp):
            for v in op.bindings.values():
                t = v.removeprefix("tensors.")
                if _is_initial_carry_candidate(t, classification, parameter_names, layer0_external_inputs):
                    layer0_external_inputs.append(t)
        elif isinstance(op, _AliasOp):
            if _is_initial_carry_candidate(op.src, classification, parameter_names, layer0_external_inputs):
                layer0_external_inputs.append(op.src)

    layer1_inputs_from_layer0: list[str] = []
    if hint.num_layers > 1:
        for op in live_ops:
            name = _op_name(op)
            if classification.get(name) != "layer.1":
                continue
            if isinstance(op, _DispatchOp):
                for v in op.bindings.values():
                    t = v.removeprefix("tensors.")
                    if classification.get(t) == "layer.0" and t not in layer1_inputs_from_layer0:
                        layer1_inputs_from_layer0.append(t)
            elif isinstance(op, _AliasOp):
                if classification.get(op.src) == "layer.0" and op.src not in layer1_inputs_from_layer0:
                    layer1_inputs_from_layer0.append(op.src)

    carry_output = _select_carry_output(
        layer1_inputs_from_layer0=layer1_inputs_from_layer0,
        layer0_external_inputs=layer0_external_inputs,
        tensor_metas=tensor_metas,
    )
    carry_inputs: list[str] = []
    if carry_output is not None:
        carry_meta = tensor_metas.get(carry_output)
        for candidate in layer0_external_inputs:
            if carry_meta is None or _same_tensor_contract(tensor_metas.get(candidate), carry_meta):
                carry_inputs.append(candidate)
                break

    pre_intermediates = {_op_name(op) for op in pre_ops}
    for op in layer0_ops:
        if isinstance(op, _DispatchOp):
            for v in op.bindings.values():
                t = v.removeprefix("tensors.")
                if t in pre_intermediates and t not in carry_inputs:
                    carry_inputs.append(t)
        elif isinstance(op, _AliasOp):
            if op.src in pre_intermediates and op.src not in carry_inputs:
                carry_inputs.append(op.src)

    return _LoopAnalysis(
        pre_ops=pre_ops,
        layer_ops=layer0_ops,
        post_ops=post_ops,
        layer_local_tensors=frozenset(layer_local),
        carry_inputs=carry_inputs,
        carry_output=carry_output,
    )


def _is_initial_carry_candidate(
    tensor_name: str,
    classification: dict[str, str],
    parameter_names: frozenset[str],
    seen: list[str],
) -> bool:
    if tensor_name in seen or tensor_name in parameter_names:
        return False
    return not classification.get(tensor_name, "").startswith("layer.")


def _select_carry_output(
    *,
    layer1_inputs_from_layer0: list[str],
    layer0_external_inputs: list[str],
    tensor_metas: dict[str, _TensorMeta],
) -> str | None:
    for output_name in layer1_inputs_from_layer0:
        output_meta = tensor_metas.get(output_name)
        if output_meta is None:
            continue
        for input_name in layer0_external_inputs:
            if _same_tensor_contract(tensor_metas.get(input_name), output_meta):
                return output_name
    return layer1_inputs_from_layer0[0] if layer1_inputs_from_layer0 else None


def _same_tensor_contract(lhs: _TensorMeta | None, rhs: _TensorMeta | None) -> bool:
    if lhs is None or rhs is None:
        return False
    return lhs.shape == rhs.shape and lhs.dtype == rhs.dtype


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


def _parameter_names(sig) -> frozenset[str]:
    names: set[str] = set()
    for spec in sig.input_specs:
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
            names.add(spec.arg.name)
    return frozenset(names)


def _graph_tensor_metas(graph: Graph, sig) -> dict[str, _TensorMeta]:
    input_kinds = {
        spec.arg.name: _TensorKind.PARAMETER
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
        else _TensorKind.USER_INPUT
        for spec in sig.input_specs
    }
    tensor_metas: dict[str, _TensorMeta] = {}
    for node in graph.nodes:
        tm = node.meta.get("tensor_meta")
        if tm is None:
            continue
        tensor_metas[node.name] = _TensorMeta(
            shape=tuple(int(d) for d in tm.shape),
            dtype=str(tm.dtype).removeprefix("torch."),
            kind=input_kinds.get(node.name, _TensorKind.INTERMEDIATE),
        )
    return tensor_metas


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
    analysis = _analyze_loop(
        graph,
        live_ops,
        classification,
        hint,
        _graph_tensor_metas(graph, prog.graph_signature),
        _parameter_names(prog.graph_signature),
    )

    # Layer params are also layer-local (they're placeholders, not in nn_module_stack)
    layer_params = _layer_param_names(prog.graph_signature, weight_prefix)
    layer_local = analysis.layer_local_tensors | layer_params

    shader_imports: dict[str, str] = {}
    carry_set = frozenset(analysis.carry_inputs)
    post_carry_targets = _post_carry_targets(analysis.post_ops, classification)
    uses_carry_in_dispatch = _uses_carry_in_dispatch(
        analysis=analysis,
        carry_set=carry_set,
        post_carry_targets=post_carry_targets,
    )

    def _tensor_ref(tensor_name: str, in_loop: bool) -> str:
        if in_loop and tensor_name in carry_set:
            return "carry"
        if in_loop and tensor_name in layer_local:
            return f"layer_t.{tensor_name}"
        return f"tensors.{tensor_name}"

    def _emit_op(op: _Op, in_loop: bool, indent: str) -> str:
        if isinstance(op, _AliasOp):
            return ""
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
        line = _emit_op(op, False, "    ")
        if line:
            lines.append(line)

    # Initialize carry
    if uses_carry_in_dispatch and analysis.carry_inputs:
        carry_init = analysis.carry_inputs[0]
        lines.append(f"    carry = tensors.{carry_init}")

    # Loop
    lines.append("    for layer_t in tensors.layers:")
    loop_body_lines = 0
    for op in analysis.layer_ops:
        line = _emit_op(op, True, "        ")
        if line:
            lines.append(line)
            loop_body_lines += 1
    if uses_carry_in_dispatch and analysis.carry_output:
        lines.append(f"        carry = layer_t.{analysis.carry_output}")
        loop_body_lines += 1
    if loop_body_lines == 0:
        lines.append("        pass")

    for op in analysis.post_ops:
        if isinstance(op, _AliasOp):
            continue
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


def _post_carry_targets(
    post_ops: list[_Op],
    classification: dict[str, str],
) -> set[str]:
    targets: set[str] = set()
    for op in post_ops:
        if isinstance(op, _DispatchOp):
            for value in op.bindings.values():
                tensor_name = value.removeprefix("tensors.")
                if classification.get(tensor_name, "").startswith("layer."):
                    targets.add(tensor_name)
        elif isinstance(op, _AliasOp):
            if classification.get(op.src, "").startswith("layer."):
                targets.add(op.src)
    return targets


def _uses_carry_in_dispatch(
    *,
    analysis: _LoopAnalysis,
    carry_set: frozenset[str],
    post_carry_targets: set[str],
) -> bool:
    for op in analysis.layer_ops:
        if isinstance(op, _DispatchOp):
            for value in op.bindings.values():
                if value.removeprefix("tensors.") in carry_set:
                    return True
    for op in analysis.post_ops:
        if isinstance(op, _DispatchOp):
            for value in op.bindings.values():
                if value.removeprefix("tensors.") in post_carry_targets:
                    return True
    return False


def generate_looped_tensor_class_sources(
    prog: ExportedProgram,
    *,
    parent_class_name: str,
    layer_class_name: str,
    parent_function_name: str,
    layer_function_name: str,
    weight_prefix: str = "",
    hint: LayerLoopHint,
    registry: ShaderRegistry = DEFAULT_REGISTRY,
) -> tuple[TensorClassContext, TensorClassContext]:
    """Generate parent + layer tensor classes.

    Returns (parent_class_context, layer_class_context).
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
    analysis = _analyze_loop(
        graph,
        live_ops,
        classification,
        hint,
        tensors,
        frozenset(param_map),
    )

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
        analysis=analysis,
        classification=classification,
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
) -> TensorClassContext:
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
                "checkpoint_key_expr": f'f"{name_template}"',
                "reference_key_expr": "None",
                "dtype_source": repr(dtype),
                "shape_source": repr(meta.shape),
                "role": "TensorRole.WEIGHT",
                "memory": "MemoryClass.MODEL_WEIGHT",
                "lifetime": "TensorLifetime.MODEL",
            })
        else:
            tensor_entries.append({
                "name": name,
                "name_source": repr(name),
                "checkpoint_key_expr": "None",
                "reference_key_expr": repr(name),
                "dtype_source": repr(dtype),
                "shape_source": repr(meta.shape),
                "role": "TensorRole.ACTIVATION",
                "memory": "MemoryClass.FRAME_WORKSPACE",
                "lifetime": "TensorLifetime.FRAME",
            })

    output_name = next(
        (n for n in reversed(tensors) if tensors[n].kind == _TensorKind.INTERMEDIATE),
        next(iter(tensors), "unknown"),
    )
    output_const = function_name.removeprefix("create_").upper() + "_OUTPUT"
    fields = tuple(tensors.keys())

    return render_tensor_class(
        class_name=class_name,
        fields=fields,
        output_const=output_const,
        output_name_source=repr(output_name),
        signature=_tensor_factory_signature(
            function_name,
            class_name,
            fields=fields,
            layered=True,
        ),
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
    analysis: _LoopAnalysis,
    classification: dict[str, str],
) -> TensorClassContext:
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
                "checkpoint_key_expr": f'"{param_map[name]}"',
                "reference_key_expr": "None",
                "dtype_source": repr(dtype),
                "shape_source": repr(meta.shape),
                "role": "TensorRole.WEIGHT",
                "memory": "MemoryClass.MODEL_WEIGHT",
                "lifetime": "TensorLifetime.MODEL",
            })
        elif kind == _TensorKind.USER_INPUT:
            tensor_entries.append({
                "name": name,
                "name_source": repr(name),
                "checkpoint_key_expr": "None",
                "reference_key_expr": "None",
                "dtype_source": repr(dtype),
                "shape_source": repr(meta.shape),
                "role": "TensorRole.INPUT",
                "memory": "MemoryClass.HOST_INPUT",
                "lifetime": "TensorLifetime.FRAME",
            })
        else:
            tensor_entries.append({
                "name": name,
                "name_source": repr(name),
                "checkpoint_key_expr": "None",
                "reference_key_expr": repr(name),
                "dtype_source": repr(dtype),
                "shape_source": repr(meta.shape),
                "role": "TensorRole.ACTIVATION",
                "memory": "MemoryClass.FRAME_WORKSPACE",
                "lifetime": "TensorLifetime.FRAME",
            })

    if output_name is None:
        output_name = next(
            (n for n in reversed(tensors) if tensors[n].kind == _TensorKind.INTERMEDIATE),
            next(iter(tensors), "unknown"),
        )

    output_const = function_name.removeprefix("create_").upper() + "_OUTPUT"
    fields = tuple(tensors.keys())

    return render_tensor_class(
        class_name=class_name,
        fields=fields,
        extra_fields=(f"layers: list[{layer_class_name}]",),
        output_const=output_const,
        output_name_source=repr(output_name),
        signature=_tensor_factory_signature(
            function_name,
            class_name,
            fields=fields,
            layered=False,
        ),
        tensors=tensor_entries,
        extra_initializers=(
            f"layers=[{layer_function_name}(prefix, layer_idx=i) for i in range({num_layers})]",
        ),
        alias_binding_lines=_loop_alias_binding_lines(analysis, classification),
    )


def _loop_alias_binding_lines(
    analysis: _LoopAnalysis,
    classification: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    carry_set = frozenset(analysis.carry_inputs)

    def tensor_ref(tensor_name: str, in_loop: bool) -> str:
        if in_loop and tensor_name in carry_set:
            return "_alias_carry"
        if in_loop and (
            tensor_name in analysis.layer_local_tensors
            or classification.get(tensor_name, "").startswith("layer.")
        ):
            return f"layer_t.{tensor_name}"
        return f"tensors.{tensor_name}"

    for op in analysis.pre_ops:
        if isinstance(op, _AliasOp):
            lines.append(
                f"    _bind_alias_source(tensors.{op.src}, tensors.{op.dst})"
            )

    layer_aliases = [op for op in analysis.layer_ops if isinstance(op, _AliasOp)]
    post_aliases = [op for op in analysis.post_ops if isinstance(op, _AliasOp)]
    if layer_aliases or post_aliases:
        if analysis.carry_inputs:
            lines.append(f"    _alias_carry = tensors.{analysis.carry_inputs[0]}")
        lines.append("    for layer_t in tensors.layers:")
        if layer_aliases:
            for op in layer_aliases:
                lines.append(
                    "        _bind_alias_source("
                    f"{tensor_ref(op.src, True)}, {tensor_ref(op.dst, True)})"
                )
        else:
            lines.append("        pass")
        if analysis.carry_output is not None:
            lines.append(f"        _alias_carry = layer_t.{analysis.carry_output}")

    for op in post_aliases:
        src = (
            "_alias_carry"
            if classification.get(op.src, "").startswith("layer.")
            else f"tensors.{op.src}"
        )
        lines.append(f"    _bind_alias_source({src}, tensors.{op.dst})")

    return lines
