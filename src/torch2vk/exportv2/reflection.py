"""Runtime reflection helpers for PyTorch module trees."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TorchModuleReflection:
    module_types: Mapping[str, str]
    parameter_shapes: Mapping[str, tuple[int, ...]]

    def require_parameter(self, name: str) -> tuple[int, ...]:
        try:
            return self.parameter_shapes[name]
        except KeyError as exc:
            raise RuntimeError(f"reflected model is missing parameter {name!r}") from exc

    def require_module_type(self, name: str) -> str:
        try:
            return self.module_types[name]
        except KeyError as exc:
            raise RuntimeError(f"reflected model is missing module {name!r}") from exc


def reflect_torch_module(module: torch.nn.Module) -> TorchModuleReflection:
    """Collect module type names and parameter shapes from an instantiated PyTorch module."""
    module_types = {name: value.__class__.__name__ for name, value in module.named_modules()}
    parameter_shapes = {
        name: tuple(int(dim) for dim in value.shape) for name, value in module.named_parameters()
    }
    return TorchModuleReflection(
        module_types=module_types,
        parameter_shapes=parameter_shapes,
    )


def instantiate_torch_module_on_meta(factory: Callable[[], torch.nn.Module]) -> torch.nn.Module:
    """Instantiate a PyTorch module on the meta device to avoid allocating real weights."""
    with torch.device("meta"):
        return factory()
