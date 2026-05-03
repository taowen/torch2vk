#!/usr/bin/env python3
"""Run the recorded Qwen3 prefill dispatch sequence through Vulkan."""

from __future__ import annotations

import importlib
import pkgutil
import struct
from dataclasses import fields, is_dataclass
from pathlib import Path

from torch2vk.logical import LogicalTensor
from torch2vk.models.qwen3_safetensor.debug import qwen3_prefill_initial_tensors
from torch2vk.models.qwen3_safetensor.execution import (
    qwen3_execution_tensors,
    record_qwen3_prefill,
)
from torch2vk.models.qwen3_safetensor.schema import qwen3_weight_tensors
from torch2vk.models.qwen3_safetensor.spec import Qwen3Spec
from torch2vk.shader import DispatchTarget, ShaderVariant
from torch2vk.storage import bind_storage, plan_storage, tensor_nbytes
from torch2vk.validation import validate_dispatch_read_write_chain
from torch2vk.vulkan_backend import VulkanBuffer, VulkanContext, create_compute_context
from torch2vk.vulkan_runner import (
    VulkanSequenceRunner,
    allocate_storage_buffers,
    read_bound_tensor_bytes,
    write_bound_tensor_bytes,
)


PACKAGE = "torch2vk.models.qwen3_safetensor.shaders"
SHADER_DIR = Path("build/shaders/qwen3_safetensor")


def main() -> int:
    spec = Qwen3Spec(
        model_type="qwen3",
        vocab_size=16,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )
    execution_tensors = qwen3_execution_tensors(batch=1, steps=1, spec=spec, max_seq_len=1)
    dispatch_target = DispatchTarget()
    record_qwen3_prefill(dispatch_target, spec=spec, tensors=execution_tensors)
    validate_dispatch_read_write_chain(
        dispatch_target.records,
        initial_tensors=qwen3_prefill_initial_tensors(spec=spec, tensors=execution_tensors),
    ).raise_for_issues()

    unbound = (*_collect_logical_tensors(execution_tensors), *qwen3_weight_tensors(spec))
    plan = plan_storage(unbound, allocation_id="qwen3-prefill")
    bound = bind_storage(unbound, plan)
    tensors = _tensor_lookup(bound)

    context = create_compute_context()
    try:
        allocations = allocate_storage_buffers(context, plan)
        resources = _resource_buffers(context)
        try:
            _write_initial_tensors(
                tensors=tensors,
                allocations=allocations,
                initial=qwen3_prefill_initial_tensors(spec=spec, tensors=execution_tensors),
            )
            runner = VulkanSequenceRunner(
                context=context,
                shader_dir=SHADER_DIR,
                variants=_shader_variants(),
            )
            runner.run_bound_storage(
                tuple(dispatch_target.records),
                tensors=tensors,
                allocations=allocations,
                resource_buffers=resources,
            )
            output_tensor = _first_tensor(tensors["output.next_token_id"])
            output_bytes = read_bound_tensor_bytes(output_tensor, allocations)
            print(
                "qwen3_prefill_dispatch=ok "
                f"dispatches={len(dispatch_target.records)} readback_bytes={len(output_bytes)}"
            )
        finally:
            for by_shader in resources.values():
                for buffer in by_shader.values():
                    buffer.close()
            for allocation in allocations.values():
                allocation.close()
    finally:
        context.close()
    return 0


def _write_initial_tensors(
    *,
    tensors: dict[str, LogicalTensor],
    allocations: dict[str, VulkanBuffer],
    initial: tuple[LogicalTensor, ...],
) -> None:
    for tensor in initial:
        bound_tensor = _first_tensor(tensors[tensor.name])
        if tensor.name == "input.input_ids":
            payload = struct.pack("<i", 0)
        elif tensor.name == "input.position_ids":
            payload = struct.pack("<i", 0)
        elif tensor.name == "input.row_indices":
            payload = struct.pack("<q", 0)
        else:
            payload = bytes(tensor_nbytes(bound_tensor))
        write_bound_tensor_bytes(bound_tensor, allocations, payload)


def _resource_buffers(context: VulkanContext) -> dict[str, dict[str, VulkanBuffer]]:
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


def _collect_logical_tensors(value: object) -> tuple[LogicalTensor, ...]:
    found: list[LogicalTensor] = []
    _collect(value, found)
    return tuple(found)


def _collect(value: object, found: list[LogicalTensor]) -> None:
    if isinstance(value, LogicalTensor):
        found.append(value)
        return
    if isinstance(value, tuple):
        for item in value:
            _collect(item, found)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect(getattr(value, field.name), found)


def _tensor_lookup(
    tensors: tuple[LogicalTensor, ...],
) -> dict[str, LogicalTensor | tuple[LogicalTensor, ...]]:
    lookup: dict[str, list[LogicalTensor]] = {}
    for tensor in tensors:
        lookup.setdefault(tensor.name, []).append(tensor)
    return {
        name: values[0] if len(values) == 1 else tuple(values)
        for name, values in lookup.items()
    }


def _first_tensor(value: LogicalTensor | tuple[LogicalTensor, ...]) -> LogicalTensor:
    return value if isinstance(value, LogicalTensor) else value[0]


def _shader_variants() -> dict[str, ShaderVariant]:
    package = importlib.import_module(PACKAGE)
    variants: dict[str, ShaderVariant] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{PACKAGE}.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, ShaderVariant):
                variants[value.name] = value
    return variants


if __name__ == "__main__":
    raise SystemExit(main())
