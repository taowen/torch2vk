#!/usr/bin/env python3
"""Create Vulkan pipeline layouts from Qwen3 shader contracts."""

from __future__ import annotations

import importlib
import pkgutil

from torch2vk.shader import ShaderVariant
from torch2vk.vulkan_backend import create_compute_context


PACKAGE = "torch2vk.models.qwen3_safetensor.shaders"


def main() -> int:
    context = create_compute_context()
    try:
        for variant in _shader_variants():
            descriptor_layout = context.create_descriptor_set_layout(variant.contract)
            try:
                pipeline_layout = context.create_pipeline_layout(
                    variant.contract,
                    descriptor_layout,
                )
                try:
                    print(f"pipeline_layout=ok {variant.name}")
                finally:
                    pipeline_layout.close()
            finally:
                descriptor_layout.close()
    finally:
        context.close()
    return 0


def _shader_variants() -> tuple[ShaderVariant, ...]:
    package = importlib.import_module(PACKAGE)
    variants: list[ShaderVariant] = []
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{PACKAGE}.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, ShaderVariant):
                variants.append(value)
    return tuple(sorted(variants, key=lambda item: item.name))


if __name__ == "__main__":
    raise SystemExit(main())
