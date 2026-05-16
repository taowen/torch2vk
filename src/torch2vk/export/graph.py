"""Export a PyTorch submodule via torch.export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal

import torch
from torch.fx import Graph, Node

KVCachePhase = Literal["prefill", "decode"]


@dataclass(frozen=True, slots=True)
class KVCacheExportHint:
    phase: KVCachePhase
    key_cache: str
    value_cache: str
    cache_position: str


@dataclass(frozen=True, slots=True)
class KVCacheInjectHint:
    """Hint for injecting KV cache ops into a graph exported without past_key_values.

    Used when exporting upstream modules directly (past_key_values=None).
    The graph has pure attention (SDPA on current k/v). This hint tells the
    framework to inject index_copy (cache write) and modify SDPA for decode.
    """

    phase: KVCachePhase
    max_seq_len: int


@dataclass(frozen=True, slots=True)
class LayerLoopHint:
    """Hint for codegen to split an unrolled graph into pre-loop / loop-body / post-loop.

    When a Module contains a for-loop over layers (e.g. nn.ModuleList),
    torch.export unrolls it. This hint tells codegen to recognize the repeating
    pattern and generate a loop dispatch instead of inlining all N copies.
    """

    layer_prefix: str
    num_layers: int


ALIAS_OPS = frozenset(
    {
        "aten.view.default",
        "aten.unsqueeze.default",
        "aten.reshape.default",
        "aten.contiguous.default",
        "aten._assert_tensor_metadata.default",
    }
)

SKIP_OPS = frozenset(
    {
        "aten._assert_tensor_metadata.default",
    }
)


def export_submodule(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    kv_cache: KVCacheExportHint | None = None,
    strict: bool = False,
) -> torch.export.ExportedProgram:
    prog = torch.export.export(module, args, kwargs=kwargs, strict=strict)
    if kv_cache is not None:
        _annotate_kv_cache(prog, kv_cache)
    return prog


def is_alias_op(node: Node) -> bool:
    target = str(node.target)
    if target in ALIAS_OPS:
        return True
    if target == "aten.to.dtype":
        return _is_float_dtype(_node_input_dtype(node)) and _is_float_dtype(_node_dtype(node))
    if target == "aten.to.dtype_layout":
        return _node_input_dtype(node) == _node_dtype(node)
    return False


def _is_float_dtype(dtype: str) -> bool:
    return dtype in {"float16", "bfloat16", "float32", "float64"}


def node_input_names(node: Node) -> tuple[str, ...]:
    override = node.meta.get("torch2vk_shader_inputs")
    if isinstance(override, tuple):
        return tuple(str(name) for name in override)

    names: list[str] = []
    for arg in node.args:
        if isinstance(arg, Node):
            names.append(arg.name)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, Node):
                    names.append(item.name)
    for value in node.kwargs.values():
        if isinstance(value, Node):
            names.append(value.name)
    return tuple(names)


def graph_output_names(graph: Graph) -> list[str]:
    names: list[str] = []
    for node in graph.nodes:
        if node.op == "output":
            _collect_output_node_names(node.args, names)
    return names


def _collect_output_node_names(value, names: list[str]) -> None:
    if isinstance(value, Node):
        names.append(value.name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_output_node_names(item, names)


def _node_dtype(node: Node) -> str:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
        return ""
    return str(tensor_meta.dtype).removeprefix("torch.")


def _node_input_dtype(node: Node) -> str:
    if node.args:
        first_arg = node.args[0]
        if isinstance(first_arg, Node):
            return _node_dtype(first_arg)
    return ""


def _annotate_kv_cache(prog: torch.export.ExportedProgram, hint: KVCacheExportHint) -> None:
    graph = prog.graph_module.graph
    _validate_kv_cache_placeholders(graph, hint)
    key_write = _find_kv_cache_write(
        graph,
        cache_name=hint.key_cache,
        cache_position_name=hint.cache_position,
        cache_kind="key",
        phase=hint.phase,
    )
    value_write = _find_kv_cache_write(
        graph,
        cache_name=hint.value_cache,
        cache_position_name=hint.cache_position,
        cache_kind="value",
        phase=hint.phase,
    )
    if hint.phase == "decode":
        _annotate_decode_cache_attention(
            graph,
            key_write=key_write,
            value_write=value_write,
            cache_position_name=hint.cache_position,
        )


def _validate_kv_cache_placeholders(graph: Graph, hint: KVCacheExportHint) -> None:
    placeholders = {node.name for node in graph.nodes if node.op == "placeholder"}
    expected = {hint.key_cache, hint.value_cache, hint.cache_position}
    missing = sorted(expected - placeholders)
    if missing:
        raise ValueError(f"KV cache hint references missing placeholders: {missing}")


def _find_kv_cache_write(
    graph: Graph,
    *,
    cache_name: str,
    cache_position_name: str,
    cache_kind: Literal["key", "value"],
    phase: KVCachePhase,
) -> Node:
    for node in graph.nodes:
        if node.op != "call_function" or str(node.target) != "aten.index_copy.default":
            continue
        if _node_arg_name(node, 0) != cache_name:
            continue
        if _node_arg_int(node, 1) != 2:
            continue
        if _node_arg_name(node, 2) != cache_position_name:
            continue
        node.meta["torch2vk_kv_cache"] = f"{phase}_{cache_kind}_write"
        if phase == "decode":
            src_name = _node_arg_name(node, 3)
            if src_name is None:
                raise ValueError(f"{node.name} decode KV cache write is missing source input")
            node.meta["torch2vk_shader_inputs"] = (cache_name, src_name)
        return node
    raise ValueError(
        f"KV cache hint did not match an index_copy write for {cache_kind}_cache={cache_name!r}"
    )


def _annotate_decode_cache_attention(
    graph: Graph,
    *,
    key_write: Node,
    value_write: Node,
    cache_position_name: str,
) -> None:
    for node in graph.nodes:
        if (
            node.op != "call_function"
            or str(node.target) != "aten.scaled_dot_product_attention.default"
        ):
            continue
        if _node_arg_name(node, 1) != key_write.name:
            continue
        if _node_arg_name(node, 2) != value_write.name:
            continue
        query_name = _node_arg_name(node, 0)
        if query_name is None:
            raise ValueError(f"{node.name} decode cache attention is missing query input")
        node.meta["torch2vk_kv_cache"] = "decode_attention"
        node.meta["torch2vk_shader_inputs"] = (
            query_name,
            key_write.name,
            value_write.name,
        )
        cache_position_node = _find_node_by_name(graph, cache_position_name)
        node.meta["torch2vk_cache_position_dtype"] = _node_dtype(cache_position_node)
        return
    raise ValueError("KV cache hint did not match decode scaled_dot_product_attention")


def _find_node_by_name(graph: Graph, name: str) -> Node:
    for node in graph.nodes:
        if node.name == name:
            return node
    raise ValueError(f"Graph is missing node {name!r}")


def _node_arg_name(node: Node, index: int) -> str | None:
    if len(node.args) <= index:
        return None
    arg = node.args[index]
    if isinstance(arg, Node):
        return arg.name
    return None


def _node_arg_int(node: Node, index: int) -> int | None:
    if len(node.args) <= index:
        return None
    arg = node.args[index]
    if isinstance(arg, int):
        return arg
    return None


def inject_kv_cache(prog: torch.export.ExportedProgram, hint: KVCacheInjectHint) -> None:
    """Inject KV cache ops into a graph exported with past_key_values=None.

    Adds key_cache, value_cache, cache_position placeholders and index_copy nodes.
    For decode phase, also modifies SDPA to use the full cache.
    After injection, calls _annotate_kv_cache so codegen handles it correctly.
    """
    from torch.export.graph_signature import InputKind, InputSpec, TensorArgument

    graph = prog.graph_module.graph

    sdpa_node = _find_sdpa_node(graph)
    k_node = sdpa_node.args[1]
    v_node = sdpa_node.args[2]
    if not isinstance(k_node, Node) or not isinstance(v_node, Node):
        raise TypeError("SDPA key/value args must be graph nodes for KV cache injection")
    k_meta = k_node.meta["tensor_meta"]

    batch, num_kv_heads, seq_len, head_dim = k_meta.shape
    cache_shape = (batch, num_kv_heads, hint.max_seq_len, head_dim)
    cache_position_shape = (seq_len,) if hint.phase == "prefill" else (1,)

    last_placeholder = _find_last_placeholder(graph)

    def _make_tensor_meta(shape, dtype):
        from torch.fx.passes.shape_prop import TensorMetadata

        return TensorMetadata(
            shape=torch.Size(shape),
            dtype=dtype,
            requires_grad=False,
            stride=tuple(range(len(shape), 0, -1)),
            memory_format=None,
            is_quantized=False,
            qparams={},
        )

    cache_dtype = torch.float16

    with graph.inserting_after(last_placeholder):
        key_cache_node = graph.placeholder("key_cache")
    key_cache_node.meta["tensor_meta"] = _make_tensor_meta(cache_shape, cache_dtype)
    key_cache_node.meta["val"] = torch.empty(cache_shape, dtype=cache_dtype, device="meta")

    with graph.inserting_after(key_cache_node):
        value_cache_node = graph.placeholder("value_cache")
    value_cache_node.meta["tensor_meta"] = _make_tensor_meta(cache_shape, cache_dtype)
    value_cache_node.meta["val"] = torch.empty(cache_shape, dtype=cache_dtype, device="meta")

    with graph.inserting_after(value_cache_node):
        cache_position_node = graph.placeholder("cache_position")
    cache_position_node.meta["tensor_meta"] = _make_tensor_meta(cache_position_shape, torch.int64)
    cache_position_node.meta["val"] = torch.empty(
        cache_position_shape, dtype=torch.int64, device="meta"
    )

    with graph.inserting_before(sdpa_node):
        index_copy_key = graph.call_function(
            torch.ops.aten.index_copy.default,
            args=(key_cache_node, 2, cache_position_node, k_node),
        )
        index_copy_key.name = "index_copy"
        index_copy_key.meta["tensor_meta"] = _make_tensor_meta(cache_shape, cache_dtype)

        index_copy_value = graph.call_function(
            torch.ops.aten.index_copy.default,
            args=(value_cache_node, 2, cache_position_node, v_node),
        )
        index_copy_value.name = "index_copy_1"
        index_copy_value.meta["tensor_meta"] = _make_tensor_meta(cache_shape, cache_dtype)

    if hint.phase == "decode":
        sdpa_node.args = (
            sdpa_node.args[0],
            index_copy_key,
            index_copy_value,
            *sdpa_node.args[3:],
        )

    output_node = _find_output_node(graph)
    current_outputs = output_node.args[0]
    if isinstance(current_outputs, tuple):
        new_outputs = (*current_outputs, index_copy_key, index_copy_value)
    else:
        new_outputs = (current_outputs, index_copy_key, index_copy_value)
    output_node.args = (new_outputs,)

    sig = prog.graph_signature
    sig.input_specs.append(
        InputSpec(
            kind=InputKind.USER_INPUT,
            arg=TensorArgument(name="key_cache"),
            target=None,
            persistent=None,
        )
    )
    sig.input_specs.append(
        InputSpec(
            kind=InputKind.USER_INPUT,
            arg=TensorArgument(name="value_cache"),
            target=None,
            persistent=None,
        )
    )
    if hint.phase == "prefill":
        sig.input_specs.append(
            InputSpec(
                kind=InputKind.USER_INPUT,
                arg=TensorArgument(name="cache_position"),
                target=None,
                persistent=None,
            )
        )

    _annotate_kv_cache(
        prog,
        KVCacheExportHint(
            phase=hint.phase,
            key_cache="key_cache",
            value_cache="value_cache",
            cache_position="cache_position",
        ),
    )


def _find_sdpa_node(graph: Graph) -> Node:
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and str(node.target) == "aten.scaled_dot_product_attention.default"
        ):
            return node
    raise ValueError("Graph does not contain scaled_dot_product_attention")


def _find_last_placeholder(graph: Graph) -> Node:
    last = None
    for node in graph.nodes:
        if node.op == "placeholder":
            last = node
    if last is None:
        raise ValueError("Graph has no placeholder nodes")
    return last


def _find_output_node(graph: Graph) -> Node:
    for node in graph.nodes:
        if node.op == "output":
            return node
    raise ValueError("Graph has no output node")
