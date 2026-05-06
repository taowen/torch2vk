"""Helpers for reading PyTorch ``ExportedProgram`` FX graphs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

def export_torch_program(
    module: Any,
    args: tuple[Any, ...],
    *,
    kwargs: Mapping[str, object] | None = None,
    strict: bool = False,
) -> object:
    """Run ``torch.export.export`` without importing torch at package import time."""
    import torch

    return torch.export.export(
        module,
        args,
        kwargs=None if kwargs is None else dict(kwargs),
        strict=strict,
    )


def torch_ops_from_exported_program(
    exported_program: object,
    *,
    tensor_name_map: Mapping[str, str] | None = None,  # kept for compatibility
) -> tuple[object, ...]:
    """Return PyTorch FX call_function nodes directly (no torch2vk op abstraction)."""
    del tensor_name_map
    return iter_fx_call_function_nodes(exported_program)


def iter_fx_call_function_nodes(exported_program: object) -> tuple[object, ...]:
    """Return PyTorch FX call_function nodes directly with no torch2vk abstraction."""
    graph = getattr(exported_program, "graph")
    return tuple(node for node in graph.nodes if getattr(node, "op", None) == "call_function")


def input_names(args: object, kwargs: object, names: Mapping[str, str]) -> tuple[str, ...]:
    values: list[str] = []
    _collect_input_names(args, names, values)
    _collect_input_names(kwargs, names, values)
    return tuple(values)


def _collect_input_names(value: object, names: Mapping[str, str], output: list[str]) -> None:
    if _is_fx_node(value):
        output.append(mapped_node_name(value, names))
    elif isinstance(value, tuple | list):
        for item in value:
            _collect_input_names(item, names, output)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_input_names(item, names, output)


def mapped_node_name(node: object, names: Mapping[str, str]) -> str:
    name = str(getattr(node, "name"))
    return names.get(name, name)


def _is_fx_node(value: object) -> bool:
    return hasattr(value, "op") and hasattr(value, "target") and hasattr(value, "name")
