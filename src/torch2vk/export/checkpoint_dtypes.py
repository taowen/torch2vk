"""Checkpoint dtype helpers for export codegen."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar, cast

import torch

from torch2vk.checkpoints.safetensors import open_safetensors_mmap

_T = TypeVar("_T")

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "int64": torch.int64,
    "int32": torch.int32,
    "uint32": torch.uint32,
}


def read_checkpoint_dtypes(model_dir: str | Path) -> dict[str, str]:
    checkpoint = _canonical_safetensors_path(Path(model_dir))
    with open_safetensors_mmap(checkpoint) as storage:
        return {name: entry.spec.dtype for name, entry in storage.tensors.items()}


def set_module_checkpoint_dtypes(
    module: torch.nn.Module,
    *,
    weight_prefix: str,
    checkpoint_dtypes: Mapping[str, str],
) -> None:
    for name, parameter in module.named_parameters(recurse=True):
        checkpoint_key = f"{weight_prefix}{name}"
        dtype_name = checkpoint_dtypes.get(checkpoint_key)
        if dtype_name is None:
            raise KeyError(f"Checkpoint tensor {checkpoint_key!r} was not found")
        _set_parameter_dtype(module, name, _torch_dtype(dtype_name), parameter.requires_grad)

    for name, buffer in module.named_buffers(recurse=True):
        checkpoint_key = f"{weight_prefix}{name}"
        dtype_name = checkpoint_dtypes.get(checkpoint_key)
        if dtype_name is None:
            continue
        _set_buffer_dtype(module, name, _torch_dtype(dtype_name))


def module_floating_dtype(module: torch.nn.Module) -> torch.dtype | None:
    for parameter in module.parameters(recurse=True):
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in module.buffers(recurse=True):
        if buffer.is_floating_point():
            return buffer.dtype
    return None


def cast_floating_tensors(value: _T, dtype: torch.dtype) -> _T:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return cast(_T, value.to(dtype=dtype))
        return value
    if isinstance(value, tuple):
        return cast(_T, tuple(cast_floating_tensors(item, dtype) for item in value))
    if isinstance(value, list):
        return cast(_T, [cast_floating_tensors(item, dtype) for item in value])
    if isinstance(value, dict):
        return cast(_T, {key: cast_floating_tensors(item, dtype) for key, item in value.items()})
    return value


def _set_parameter_dtype(
    module: torch.nn.Module,
    name: str,
    dtype: torch.dtype,
    requires_grad: bool,
) -> None:
    parent, leaf = _lookup_parent_module(module, name)
    parameter = parent._parameters[leaf]
    if parameter is None:
        return
    if parameter.dtype == dtype:
        return
    converted = torch.nn.Parameter(parameter.detach().to(dtype=dtype), requires_grad=requires_grad)
    parent._parameters[leaf] = converted


def _set_buffer_dtype(module: torch.nn.Module, name: str, dtype: torch.dtype) -> None:
    parent, leaf = _lookup_parent_module(module, name)
    buffer = parent._buffers[leaf]
    if buffer is None:
        return
    if buffer.dtype == dtype:
        return
    parent._buffers[leaf] = buffer.to(dtype=dtype)


def _lookup_parent_module(module: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = parent.get_submodule(part)
    return parent, parts[-1]


def _torch_dtype(dtype: str) -> torch.dtype:
    result = _TORCH_DTYPES.get(dtype)
    if result is None:
        raise ValueError(f"Unsupported checkpoint dtype: {dtype}")
    return result


def _canonical_safetensors_path(model_dir: Path) -> Path:
    primary = model_dir / "model.safetensors"
    if primary.exists():
        return primary
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return index
    candidates = sorted(model_dir.glob("*.safetensors")) + sorted(
        model_dir.glob("*.safetensors.index.json")
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(f"Could not find a safetensors checkpoint in {model_dir}")
    raise RuntimeError(
        f"Found multiple safetensors checkpoints in {model_dir}; "
        "use model.safetensors or model.safetensors.index.json as the canonical checkpoint"
    )
