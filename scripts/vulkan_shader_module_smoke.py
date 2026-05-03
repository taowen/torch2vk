#!/usr/bin/env python3
"""Create Vulkan shader modules from compiled Qwen3 SPIR-V files."""

from __future__ import annotations

from pathlib import Path

from torch2vk.vulkan_backend import create_compute_context


def main() -> int:
    shader_dir = Path("build/shaders/qwen3_safetensor")
    spirv_paths = sorted(shader_dir.glob("*.spv"))
    if not spirv_paths:
        raise FileNotFoundError(f"No compiled Qwen3 SPIR-V files in {shader_dir}")

    context = create_compute_context()
    try:
        for path in spirv_paths:
            module = context.create_shader_module(path.read_bytes())
            try:
                print(f"shader_module=ok {path.name}")
            finally:
                module.close()
    finally:
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
