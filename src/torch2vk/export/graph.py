"""Export a PyTorch submodule via torch.export."""

from __future__ import annotations

from typing import Any

import torch
from torch.fx import Node

ALIAS_OPS = frozenset({
    "aten.view.default",
    "aten.unsqueeze.default",
    "aten.reshape.default",
    "aten.contiguous.default",
    "aten._assert_tensor_metadata.default",
})

SKIP_OPS = frozenset({
    "aten._assert_tensor_metadata.default",
})


def export_submodule(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    strict: bool = False,
) -> torch.export.ExportedProgram:
    return torch.export.export(module, args, kwargs=kwargs, strict=strict)


def is_alias_op(node: Node) -> bool:
    target = str(node.target)
    if target in ALIAS_OPS:
        return True
    if target == "aten.to.dtype":
        return _node_input_dtype(node) == _node_dtype(node)
    return False


def node_input_names(node: Node) -> tuple[str, ...]:
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
