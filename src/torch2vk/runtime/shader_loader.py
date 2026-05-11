"""Dynamic loading for generated shader packages."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from torch2vk.runtime.shader import ShaderVariant

_SHADER_CACHE: dict[tuple[str, str], ShaderVariant] = {}


def load_shader(shader_package: str, name: str) -> ShaderVariant:
    key = (shader_package, name)
    cached = _SHADER_CACHE.get(key)
    if cached is not None:
        return cached

    module = importlib.import_module(f"{shader_package}.{name.lower()}")
    value = getattr(module, name.upper())
    if not isinstance(value, ShaderVariant):
        raise TypeError(f"{shader_package}.{name.lower()}.{name.upper()} is not a ShaderVariant")
    _SHADER_CACHE[key] = value
    return value


def make_shader_loader(shader_package: str) -> Callable[[str], ShaderVariant]:
    def get_shader(name: str) -> ShaderVariant:
        return load_shader(shader_package, name)

    return get_shader
