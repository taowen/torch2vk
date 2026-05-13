"""Dynamic loading for generated shader packages."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

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


def make_shader_loader(
    shader_package: str,
    *,
    extra_variants: Sequence[ShaderVariant] = (),
) -> Callable[[str], ShaderVariant]:
    extra_by_name = {variant.name: variant for variant in extra_variants}

    def get_shader(name: str) -> ShaderVariant:
        variant = extra_by_name.get(name)
        if variant is not None:
            return variant
        return load_shader(shader_package, name)

    return get_shader
