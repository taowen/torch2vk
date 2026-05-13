"""LogicalTensor dataclass generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, TypeAlias, cast

from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import Graph, Node

from torch2vk.export._templates import render_template
from torch2vk.export.dispatch_codegen import (
    _AliasOp,
    _DispatchOp,
    _collect_ops,
    _find_graph_outputs,
    _prune_dead_ops,
    _resolve_all_variants,
)
from torch2vk.export.dtype_policy import logical_tensor_dtype, requires_float32_intermediate
from torch2vk.export.graph import SKIP_OPS
from torch2vk.export.quantization import Q4KMWeightQuantization
from torch2vk.export.registry import DEFAULT_REGISTRY, ShaderRegistry


TensorClassContext: TypeAlias = dict[str, object]


class _TensorKind(Enum):
    PARAMETER = "parameter"
    USER_INPUT = "user_input"
    INTERMEDIATE = "intermediate"


@dataclass(frozen=True, slots=True)
class _TensorMeta:
    shape: tuple[int, ...]
    dtype: str
    kind: _TensorKind
    force_float32: bool = False


def render_tensor_module(classes: list[TensorClassContext]) -> str:
    q4_k_m_config = _q4_k_m_config(classes)
    uses_request_state_outputs = any(
        bool(cls.get("uses_request_state_outputs", True))
        for cls in classes
    )
    uses_alias_bindings = any(
        bool(cls.get("alias_binding_lines"))
        for cls in classes
    )
    return (
        render_template(
            "tensor_module.py.j2",
            classes=classes,
            q4_k_m_config=q4_k_m_config,
            uses_request_state_outputs=uses_request_state_outputs,
            uses_alias_bindings=uses_alias_bindings,
            uses_q4_k_words_layout=q4_k_m_config is not None
            or _uses_layout_source(classes, "q4_k_words_layout"),
            uses_q6_k_halfwords_layout=(
                (
                    q4_k_m_config is not None
                    and bool(q4_k_m_config["q6_names"] or q4_k_m_config["q6_prefixes"])
                )
                or _uses_layout_source(classes, "q6_k_halfwords_layout")
            ),
            uses_q8_0_halfwords_layout=q4_k_m_config is not None
            or _uses_layout_source(classes, "q8_0_halfwords_layout"),
        ).rstrip()
        + "\n"
    )


def _q4_k_m_config(classes: list[TensorClassContext]) -> dict[str, tuple[str, ...]] | None:
    for cls in classes:
        config = cls.get("q4_k_m_config")
        if config is not None:
            return cast(dict[str, tuple[str, ...]], config)
    return None


def _uses_layout_source(classes: list[TensorClassContext], name: str) -> bool:
    for cls in classes:
        tensors = cast(tuple[dict[str, str], ...], cls["tensors"])
        if any(name in tensor["layout_source"] for tensor in tensors):
            return True
    return False


def render_tensor_class(
    *,
    class_name: str,
    fields,
    output_const: str | None,
    output_name_source: str | None,
    signature: str,
    tensors,
    alias_ops=(),
    extra_fields=(),
    extra_initializers=(),
    alias_binding_lines: list[str] | None = None,
    q4_k_m_config: dict[str, tuple[str, ...]] | None = None,
    uses_request_state_outputs: bool = True,
) -> TensorClassContext:
    if alias_binding_lines is None:
        alias_binding_lines = [
            f"    _bind_alias_source(tensors.{alias.src}, tensors.{alias.dst})"
            for alias in alias_ops
        ]
    return {
        "class_name": class_name,
        "fields": tuple(fields),
        "extra_fields": tuple(extra_fields),
        "output_const": output_const,
        "output_name_source": output_name_source,
        "signature": signature,
        "tensors": tuple(tensors),
        "extra_initializers": tuple(extra_initializers),
        "alias_binding_lines": tuple(alias_binding_lines),
        "q4_k_m_config": q4_k_m_config,
        "uses_request_state_outputs": uses_request_state_outputs,
    }


def _tensor_factory_signature(
    function_name: str,
    class_name: str,
    *,
    fields: tuple[str, ...],
    layered: bool,
    shape_parameters: tuple[str, ...] = (),
    uses_request_state_outputs: bool = True,
) -> str:
    params = ["prefix: str"]
    if layered:
        params.append("layer_idx: int")
    params.append("*")
    params.extend(f"{name}: int" for name in shape_parameters)
    params.extend(f"{field}: LogicalTensor | None = None" for field in fields)
    if uses_request_state_outputs:
        params.append("request_state_outputs: Collection[str] = frozenset()")
    return f"def {function_name}(\n    " + ",\n    ".join(params) + f",\n) -> {class_name}:"


def generate_tensor_class_source(
    prog: ExportedProgram,
    *,
    class_name: str = "ExportedTensors",
    function_name: str = "create_exported",
    weight_prefix: str = "",
    checkpoint: str | None = None,
    is_layered: bool | None = None,
    registry: ShaderRegistry = DEFAULT_REGISTRY,
    weight_quantization: Q4KMWeightQuantization | None = None,
    shape_exprs: Mapping[int, str] | None = None,
) -> TensorClassContext:
    """Generate tensor dataclass + factory function context for a single submodule."""
    graph = prog.graph_module.graph
    sig = prog.graph_signature

    tensors: dict[str, _TensorMeta] = {}
    user_inputs: list[str] = []
    param_map: dict[str, str] = {}

    for spec in sig.input_specs:
        for node in graph.nodes:
            if node.name == spec.arg.name:
                tm = node.meta.get("tensor_meta")
                if tm:
                    shape = tuple(int(d) for d in tm.shape)
                    dtype = _node_dtype(node)
                    is_param = spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
                    tensors[spec.arg.name] = _TensorMeta(
                        shape=shape,
                        dtype=dtype,
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
                dtype = _node_dtype(node)
                tensors[node.name] = _TensorMeta(
                    shape=shape,
                    dtype=dtype,
                    kind=_TensorKind.INTERMEDIATE,
                    force_float32=requires_float32_intermediate(node),
                )

    output_name = _find_output_name(graph, tensors)

    node_variants = _resolve_all_variants(graph, registry)
    ops = _collect_ops(graph, node_variants)
    output_names = _find_graph_outputs(graph)
    if not output_names:
        output_names = [output_name] if output_name else []
    live_ops = _prune_dead_ops(ops, output_names)
    alias_ops = tuple(op for op in live_ops if isinstance(op, _AliasOp))
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
    param_map = {k: v for k, v in param_map.items() if k in live_tensors}

    if is_layered is None:
        is_layered = any(re.search(r"\.layers\.\d+\.", v) for v in param_map.values())

    tensor_entries = []
    for name, meta in tensors.items():
        tensor_entries.append(
            _tensor_entry(
                name=name,
                meta=meta,
                checkpoint_key=_checkpoint_key_expr(
                    name=name,
                    meta=meta,
                    param_map=param_map,
                    is_layered=is_layered,
                ),
                checkpoint=repr(checkpoint) if meta.kind == _TensorKind.PARAMETER else "None",
                concrete_checkpoint_key=param_map.get(name),
                reference_key="None" if meta.kind != _TensorKind.INTERMEDIATE else repr(name),
                weight_quantization=weight_quantization,
                shape_exprs=shape_exprs,
                activation_dtype=registry.activation_dtype,
            )
        )

    output_const = function_name.removeprefix("create_").upper() + "_OUTPUT"
    fields = tuple(tensors.keys())
    shape_parameters = _shape_parameter_names(tensor_entries)
    return render_tensor_class(
        class_name=class_name,
        fields=fields,
        output_const=output_const,
        output_name_source=repr(output_name),
        signature=_tensor_factory_signature(
            function_name,
            class_name,
            fields=fields,
            layered=is_layered,
            shape_parameters=shape_parameters,
        ),
        tensors=tensor_entries,
        alias_ops=alias_ops,
        q4_k_m_config=_q4_k_m_config_from_quantization(weight_quantization),
    )


def generate_weight_tensor_class_source(
    *,
    class_name: str,
    function_name: str,
    field_name: str,
    checkpoint_key: str,
    dtype: str,
    shape: tuple[int, ...],
    weight_quantization: Q4KMWeightQuantization | None = None,
    shape_exprs: Mapping[int, str] | None = None,
) -> TensorClassContext:
    """Generate a tensor class for a standalone model weight."""
    meta = _TensorMeta(
        shape=shape,
        dtype=dtype,
        kind=_TensorKind.PARAMETER,
    )
    tensor_entry = _tensor_entry(
        name=field_name,
        meta=meta,
        checkpoint_key=repr(checkpoint_key),
        checkpoint="None",
        concrete_checkpoint_key=checkpoint_key,
        reference_key="None",
        weight_quantization=weight_quantization,
        shape_exprs=shape_exprs,
    )
    return render_tensor_class(
        class_name=class_name,
        fields=(field_name,),
        output_const=None,
        output_name_source=None,
        signature=_tensor_factory_signature(
            function_name,
            class_name,
            fields=(field_name,),
            layered=False,
            uses_request_state_outputs=False,
        ),
        tensors=(tensor_entry,),
        q4_k_m_config=_q4_k_m_config_from_quantization(weight_quantization),
        uses_request_state_outputs=False,
    )


def _q4_k_m_config_from_quantization(
    weight_quantization: Q4KMWeightQuantization | None,
) -> dict[str, tuple[str, ...]] | None:
    if weight_quantization is None:
        return None
    return {
        "q6_names": tuple(sorted(weight_quantization.q6_tensor_names)),
        "q6_prefixes": weight_quantization.q6_tensor_prefixes,
        "q8_names": tuple(sorted(weight_quantization.q8_tensor_names)),
        "q8_prefixes": weight_quantization.q8_tensor_prefixes,
    }


def _tensor_entry(
    *,
    name: str,
    meta: _TensorMeta,
    checkpoint_key: str,
    checkpoint: str,
    concrete_checkpoint_key: str | None,
    reference_key: str,
    weight_quantization: Q4KMWeightQuantization | None = None,
    shape_exprs: Mapping[int, str] | None = None,
    activation_dtype: str = "float16",
) -> dict[str, object]:
    kind = meta.kind
    dtype = logical_tensor_dtype(
        is_parameter=kind == _TensorKind.PARAMETER,
        dtype=meta.dtype,
        activation_dtype=activation_dtype,
        force_float32=meta.force_float32,
    )
    shape = meta.shape
    layout_source = "CONTIGUOUS_LAYOUT"
    spec_source = f"TensorSpec(dtype={dtype!r}, shape={_shape_source(shape, shape_exprs or {})})"
    if kind == _TensorKind.PARAMETER and weight_quantization is not None:
        if concrete_checkpoint_key is None:
            raise ValueError(f"parameter tensor {name} is missing checkpoint key")
        quantized = weight_quantization.declare(
            checkpoint_key=concrete_checkpoint_key,
            dtype=dtype,
            shape=shape,
        )
        dtype = quantized.dtype
        shape = quantized.shape
        layout_source = quantized.layout_source
        spec_source = (
            f"_quantized_weight_spec({checkpoint_key}, "
            f"dtype={logical_tensor_dtype(is_parameter=True, dtype=meta.dtype, activation_dtype=activation_dtype, force_float32=meta.force_float32)!r}, "
            f"shape={_shape_source(meta.shape, shape_exprs or {})})"
        )
        layout_source = (
            f"_quantized_weight_layout({checkpoint_key}, "
            f"dtype={logical_tensor_dtype(is_parameter=True, dtype=meta.dtype, activation_dtype=activation_dtype, force_float32=meta.force_float32)!r}, "
            f"shape={_shape_source(meta.shape, shape_exprs or {})})"
        )
    if kind == _TensorKind.PARAMETER:
        role = "TensorRole.WEIGHT"
        memory = "MemoryClass.MODEL_WEIGHT"
        lifetime = "TensorLifetime.MODEL"
    elif kind == _TensorKind.USER_INPUT:
        role = "TensorRole.INPUT"
        memory = "MemoryClass.HOST_INPUT"
        lifetime = "TensorLifetime.FRAME"
    else:
        role = "TensorRole.ACTIVATION"
        memory = "MemoryClass.FRAME_WORKSPACE"
        lifetime = "TensorLifetime.FRAME"
    return {
        "name": name,
        "name_source": repr(name),
        "checkpoint_key_expr": checkpoint_key,
        "checkpoint_expr": checkpoint,
        "reference_key_expr": reference_key,
        "dtype_source": repr(dtype),
        "shape_source": _shape_source(shape, shape_exprs or {}),
        "spec_source": spec_source,
        "shape_parameters": tuple(_shape_dim_names(shape, shape_exprs or {})),
        "layout_source": layout_source,
        "role": role,
        "memory": memory,
        "lifetime": lifetime,
    }


def _shape_source(shape: tuple[int, ...], shape_exprs: Mapping[int, str]) -> str:
    if not shape:
        return "()"
    parts = tuple(shape_exprs.get(dim, repr(dim)) for dim in shape)
    suffix = "," if len(parts) == 1 else ""
    return "(" + ", ".join(parts) + suffix + ")"


def _shape_dim_names(shape: tuple[int, ...], shape_exprs: Mapping[int, str]) -> tuple[str, ...]:
    names: list[str] = []
    for dim in shape:
        expr = shape_exprs.get(dim)
        if expr is None:
            continue
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr):
            if name not in names:
                names.append(name)
    return tuple(names)


def _shape_parameter_names(tensor_entries: list[dict[str, object]]) -> tuple[str, ...]:
    names: list[str] = []
    for entry in tensor_entries:
        for name in cast(tuple[str, ...], entry["shape_parameters"]):
            if name not in names:
                names.append(name)
    return tuple(names)


def _node_dtype(node: Node) -> str:
    tm = node.meta.get("tensor_meta")
    if tm is None:
        return ""
    return str(tm.dtype).removeprefix("torch.")


def _checkpoint_key_expr(
    *,
    name: str,
    meta: _TensorMeta,
    param_map: dict[str, str],
    is_layered: bool,
) -> str:
    if meta.kind != _TensorKind.PARAMETER:
        return "None"
    safetensors_key = param_map[name]
    if is_layered:
        name_template = re.sub(r"\.layers\.(\d+)\.", ".layers.{layer_idx}.", safetensors_key)
        return f'f"{name_template}"'
    return f'"{safetensors_key}"'


def _find_output_name(graph: Graph, tensors: dict[str, _TensorMeta]) -> str | None:
    for node in graph.nodes:
        if node.op == "output":
            names = _collect_graph_output_names(node.args, tensors)
            if names:
                return names[0]
    return None


def _collect_graph_output_names(value: object, tensors: dict[str, _TensorMeta]) -> list[str]:
    names: list[str] = []
    if isinstance(value, Node):
        if value.name in tensors:
            names.append(value.name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            names.extend(_collect_graph_output_names(item, tensors))
    return names
