"""Runtime helpers for Qwen3 safetensor Vulkan execution."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import cast

from torch2vk.logical import LogicalTensor
from torch2vk.vulkan_backend import VulkanBuffer, VulkanContext


def qwen3_collect_logical_tensors(value: object) -> tuple[LogicalTensor, ...]:
    found: list[LogicalTensor] = []
    _collect(value, found)
    return tuple(found)


def qwen3_tensor_lookup(
    tensors: tuple[LogicalTensor, ...],
) -> dict[str, LogicalTensor | tuple[LogicalTensor, ...]]:
    lookup: dict[str, list[LogicalTensor]] = {}
    for tensor in tensors:
        lookup.setdefault(tensor.name, []).append(tensor)
    return {
        name: values[0] if len(values) == 1 else tuple(values)
        for name, values in lookup.items()
    }


def qwen3_first_tensor(value: LogicalTensor | tuple[LogicalTensor, ...]) -> LogicalTensor:
    return value if isinstance(value, LogicalTensor) else value[0]


def qwen3_resource_buffers(context: VulkanContext) -> dict[str, dict[str, VulkanBuffer]]:
    linear_fuse0 = context.create_host_buffer(nbytes=4)
    linear_fuse1 = context.create_host_buffer(nbytes=4)
    add_partial = context.create_host_buffer(nbytes=4)
    return {
        "linear_bf16_f32": {
            "fuse0_placeholder": linear_fuse0,
            "fuse1_placeholder": linear_fuse1,
        },
        "add_f32_f32_f32_norepeat": {
            "partial_buffer": add_partial,
        },
    }


def qwen3_close_resource_buffers(resources: dict[str, dict[str, VulkanBuffer]]) -> None:
    for by_shader in resources.values():
        for buffer in by_shader.values():
            buffer.close()


def _collect(value: object, found: list[LogicalTensor]) -> None:
    if isinstance(value, LogicalTensor):
        found.append(value)
        return
    if isinstance(value, tuple):
        for item in cast("tuple[object, ...]", value):
            _collect(item, found)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect(getattr(value, field.name), found)
