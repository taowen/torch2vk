#!/usr/bin/env python3
"""Allocate and update Vulkan descriptor sets for Qwen3 shader contracts."""

from __future__ import annotations

import importlib
import pkgutil

from torch2vk.shader import ShaderContract, ShaderVariant
from torch2vk.vulkan_backend import VulkanBuffer, create_compute_context


PACKAGE = "torch2vk.models.qwen3_safetensor.shaders"


def main() -> int:
    context = create_compute_context()
    try:
        for variant in _shader_variants():
            layout = context.create_descriptor_set_layout(variant.contract)
            pool = context.create_descriptor_pool(variant.contract)
            buffers: list[VulkanBuffer] = []
            try:
                descriptor_set = context.allocate_descriptor_set(
                    descriptor_pool=pool,
                    descriptor_set_layout=layout,
                )
                descriptor_buffers, descriptor_types = _descriptor_buffers(
                    context,
                    variant.contract,
                    buffers,
                )
                context.update_descriptor_set(
                    descriptor_set,
                    descriptor_buffers,
                    descriptor_types=descriptor_types,
                )
                print(f"descriptor_set=ok {variant.name}")
            finally:
                for buffer in buffers:
                    buffer.close()
                pool.close()
                layout.close()
    finally:
        context.close()
    return 0


def _descriptor_buffers(
    context,
    contract: ShaderContract,
    buffers: list[VulkanBuffer],
) -> tuple[dict[int, VulkanBuffer], dict[int, str]]:
    descriptors: dict[int, VulkanBuffer] = {}
    descriptor_types: dict[int, str] = {}
    for binding in contract.bindings:
        buffer = context.create_host_buffer(nbytes=256)
        buffers.append(buffer)
        descriptors[binding.binding] = buffer
        descriptor_types[binding.binding] = binding.descriptor_type
    for resource in contract.resources:
        buffer = context.create_host_buffer(nbytes=256)
        buffers.append(buffer)
        descriptors[resource.binding] = buffer
        descriptor_types[resource.binding] = resource.descriptor_type
    for uniform in contract.uniforms:
        buffer = context.create_host_buffer(nbytes=16)
        buffers.append(buffer)
        descriptors[uniform.binding] = buffer
        descriptor_types[uniform.binding] = "uniform_buffer"
    return descriptors, descriptor_types


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
