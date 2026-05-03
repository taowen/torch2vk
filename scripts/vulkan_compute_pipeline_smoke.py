#!/usr/bin/env python3
"""Create Vulkan compute pipelines for compiled Qwen3 shaders."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from torch2vk.shader import ShaderVariant
from torch2vk.vulkan_backend import create_compute_context


PACKAGE = "torch2vk.models.qwen3_safetensor.shaders"


def main() -> int:
    shader_dir = Path("build/shaders/qwen3_safetensor")
    context = create_compute_context()
    try:
        for variant in _shader_variants():
            spirv_path = shader_dir / f"{variant.name}.spv"
            if not spirv_path.exists():
                raise FileNotFoundError(f"Missing compiled shader: {spirv_path}")
            module = context.create_shader_module(spirv_path.read_bytes())
            descriptor_layout = context.create_descriptor_set_layout(variant.contract)
            pipeline_layout = context.create_pipeline_layout(variant.contract, descriptor_layout)
            try:
                pipeline = context.create_compute_pipeline(
                    shader_module=module,
                    pipeline_layout=pipeline_layout,
                    specialization_constants=variant.specialization_constants,
                )
                try:
                    print(f"compute_pipeline=ok {variant.name}")
                finally:
                    pipeline.close()
            finally:
                pipeline_layout.close()
                descriptor_layout.close()
                module.close()
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
