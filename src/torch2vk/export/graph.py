"""Helpers for reading PyTorch ``ExportedProgram`` FX graphs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch2vk.export.ir import TorchOpPattern


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
    tensor_name_map: Mapping[str, str] | None = None,
) -> tuple[TorchOpPattern, ...]:
    """Convert call-function FX nodes into torch2vk op declarations."""
    graph = getattr(exported_program, "graph")
    names = {} if tensor_name_map is None else dict(tensor_name_map)
    ops: list[TorchOpPattern] = []
    for node in graph.nodes:
        if getattr(node, "op") != "call_function":
            continue
        output = _mapped_node_name(node, names)
        ops.append(
            TorchOpPattern(
                target=_target_name(getattr(node, "target")),
                inputs=_input_names(getattr(node, "args", ()), getattr(node, "kwargs", {}), names),
                outputs=(output,),
                name=str(getattr(node, "name")),
                op=str(getattr(node, "op")),
                args=tuple(_serialize_arg(arg, names) for arg in getattr(node, "args", ())),
                kwargs=tuple(
                    (str(key), _serialize_arg(value, names))
                    for key, value in getattr(node, "kwargs", {}).items()
                ),
                shape=_node_shape(node),
                dtype=_node_dtype(node),
            )
        )
    return tuple(ops)


def _input_names(args: object, kwargs: object, names: Mapping[str, str]) -> tuple[str, ...]:
    values: list[str] = []
    _collect_input_names(args, names, values)
    _collect_input_names(kwargs, names, values)
    return tuple(values)


def _collect_input_names(value: object, names: Mapping[str, str], output: list[str]) -> None:
    if _is_fx_node(value):
        output.append(_mapped_node_name(value, names))
    elif isinstance(value, tuple | list):
        for item in value:
            _collect_input_names(item, names, output)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_input_names(item, names, output)


def _serialize_arg(value: object, names: Mapping[str, str]) -> object:
    if _is_fx_node(value):
        return _mapped_node_name(value, names)
    if isinstance(value, tuple):
        return tuple(_serialize_arg(item, names) for item in value)
    if isinstance(value, list):
        return [_serialize_arg(item, names) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_arg(item, names) for key, item in value.items()}
    return value


def _mapped_node_name(node: object, names: Mapping[str, str]) -> str:
    name = str(getattr(node, "name"))
    return names.get(name, name)


def _target_name(target: object) -> str:
    text = str(target)
    if text.startswith("aten."):
        return text
    name = getattr(target, "__name__", "")
    namespace = getattr(getattr(target, "namespace", None), "name", "")
    overload = getattr(target, "overloadpacket", None)
    if namespace and overload is not None:
        overload_name = getattr(target, "_overloadname", "")
        base = f"{namespace}.{getattr(overload, '__name__', name)}"
        return f"{base}.{overload_name}" if overload_name else base
    return name or text


def _node_shape(node: object) -> tuple[int, ...] | None:
    val = _node_meta_value(node)
    shape = getattr(val, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _node_dtype(node: object) -> str | None:
    val = _node_meta_value(node)
    dtype = getattr(val, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _node_meta_value(node: object) -> object | None:
    meta = getattr(node, "meta", None)
    if not isinstance(meta, Mapping):
        return None
    value = meta.get("val")
    if isinstance(value, tuple):
        return value[0] if value else None
    return value


def _is_fx_node(value: object) -> bool:
    return hasattr(value, "op") and hasattr(value, "target") and hasattr(value, "name")

