"""Generated shader loader."""

from __future__ import annotations

import importlib

from torch2vk.runtime.shader import ShaderVariant

_SHADER_CACHE: dict[str, ShaderVariant] = {}


def get_shader(name: str) -> ShaderVariant:
    cached = _SHADER_CACHE.get(name)
    if cached is not None:
        return cached
    module = importlib.import_module(f"{__package__}.{name.lower()}")
    constant_name = name.upper()
    value = getattr(module, constant_name)
    if not isinstance(value, ShaderVariant):
        raise TypeError(f"{module.__name__}.{constant_name} is not a ShaderVariant")
    _SHADER_CACHE[name] = value
    return value
