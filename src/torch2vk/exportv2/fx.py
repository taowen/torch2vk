"""Direct projections from PyTorch FX nodes.

FX nodes are the exporter IR. The functions here only project a node into the
target/input/output tuple expected by lowering, or expose metadata already
stored on the node.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias, cast

import torch

from torch2vk.exportv2.protocols import ExportedProgramLike, FxNodeLike


StaticNode: TypeAlias = tuple[str, tuple[str, ...], tuple[str, ...]]
FxNodeProjector: TypeAlias = Callable[[FxNodeLike, Mapping[str, str]], StaticNode]


def export_program(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    kwargs: Mapping[str, torch.Tensor] | None = None,
    strict: bool = False,
) -> ExportedProgramLike:
    """Run ``torch.export.export`` and return its FX-backed program."""
    return cast(
        ExportedProgramLike,
        torch.export.export(
            module,
            args,
            kwargs=None if kwargs is None else dict(kwargs),
            strict=strict,
        ),
    )


def export_torch_program(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    kwargs: Mapping[str, torch.Tensor] | None = None,
    strict: bool = False,
) -> ExportedProgramLike:
    return export_program(
        module,
        args,
        kwargs=kwargs,
        strict=strict,
    )


def call_function_nodes(exported_program: ExportedProgramLike) -> tuple[FxNodeLike, ...]:
    return tuple(
        node for node in exported_program.graph.nodes if getattr(node, "op") == "call_function"
    )


def iter_fx_call_function_nodes(exported_program: ExportedProgramLike) -> tuple[FxNodeLike, ...]:
    return call_function_nodes(exported_program)


def project_fx_node(
    node: FxNodeLike,
    name_map: Mapping[str, str],
) -> StaticNode:
    return (
        str(node.target),
        fx_node_input_names(node, name_map),
        (mapped_node_name(node, name_map),),
    )


def project_fx_nodes(
    nodes: Sequence[FxNodeLike],
    *,
    name_map: Mapping[str, str] | None = None,
    project: FxNodeProjector = project_fx_node,
) -> tuple[StaticNode, ...]:
    names = {} if name_map is None else name_map
    return tuple(project(node, names) for node in nodes)


def fx_node_input_names(node: FxNodeLike, name_map: Mapping[str, str]) -> tuple[str, ...]:
    values: list[str] = []
    _collect_input_names(node.args, name_map, values)
    _collect_input_names(node.kwargs, name_map, values)
    return tuple(values)


def input_names(args: object, kwargs: object, name_map: Mapping[str, str]) -> tuple[str, ...]:
    values: list[str] = []
    _collect_input_names(args, name_map, values)
    _collect_input_names(kwargs, name_map, values)
    return tuple(values)


def mapped_node_name(node: FxNodeLike, name_map: Mapping[str, str]) -> str:
    name = str(node.name)
    return name_map.get(name, name)


def fx_node_shape(node: FxNodeLike) -> tuple[int, ...] | None:
    tensor_meta = _fx_node_tensor_meta(node)
    if tensor_meta is None:
        return None
    shape = getattr(tensor_meta, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def fx_node_dtype(node: FxNodeLike) -> str | None:
    tensor_meta = _fx_node_tensor_meta(node)
    if tensor_meta is None:
        return None
    dtype = getattr(tensor_meta, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _collect_input_names(value: object, name_map: Mapping[str, str], output: list[str]) -> None:
    if _is_fx_node(value):
        output.append(name_map.get(str(getattr(value, "name")), str(getattr(value, "name"))))
    elif isinstance(value, tuple | list):
        for item in value:
            _collect_input_names(item, name_map, output)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_input_names(item, name_map, output)


def _fx_node_tensor_meta(node: FxNodeLike) -> object | None:
    return node.meta.get("tensor_meta")


def _is_fx_node(value: object) -> bool:
    return hasattr(value, "op") and hasattr(value, "target") and hasattr(value, "name")
