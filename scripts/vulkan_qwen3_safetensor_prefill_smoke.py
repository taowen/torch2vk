#!/usr/bin/env python3
"""Run a Qwen3 safetensor layer-prefix prefill sequence through Vulkan."""

from __future__ import annotations

import importlib
import os
import pkgutil
import struct
from dataclasses import replace
from pathlib import Path

from torch2vk.logical import LogicalTensor
from torch2vk.models.qwen3_safetensor.debug import qwen3_prefill_initial_tensors
from torch2vk.models.qwen3_safetensor.execution import (
    qwen3_execution_tensors,
    record_qwen3_prefill,
)
from torch2vk.models.qwen3_safetensor.runtime import (
    qwen3_close_resource_buffers,
    qwen3_collect_logical_tensors,
    qwen3_first_tensor,
    qwen3_resource_buffers,
    qwen3_tensor_lookup,
)
from torch2vk.models.qwen3_safetensor.schema import qwen3_weight_tensors
from torch2vk.models.qwen3_safetensor.spec import load_qwen3_spec
from torch2vk.models.qwen3_safetensor.weights import qwen3_safetensor_weight_payloads
from torch2vk.shader import DispatchTarget, ShaderVariant
from torch2vk.storage import bind_storage, plan_storage, tensor_nbytes
from torch2vk.validation import validate_dispatch_read_write_chain
from torch2vk.vulkan_backend import VulkanBuffer, create_compute_context
from torch2vk.vulkan_runner import (
    VulkanSequenceRunner,
    allocate_storage_buffers,
    read_bound_tensor_bytes,
    write_bound_tensor_bytes,
    write_bound_tensor_payloads,
)


DEFAULT_MODEL_DIR = Path("models/weights/qwen3-0.6b-safetensor")
PACKAGE = "torch2vk.models.qwen3_safetensor.shaders"
SHADER_DIR = Path("build/shaders/qwen3_safetensor")


def main() -> int:
    model_dir = Path(os.environ.get("QWEN3_SAFETENSOR_DIR", str(DEFAULT_MODEL_DIR)))
    if not model_dir.exists():
        print(f"qwen3_safetensor_prefill=skip reason=missing_model_dir path={model_dir}")
        return 0
    base_spec = load_qwen3_spec(model_dir)
    layer_count = int(os.environ.get("QWEN3_PREFILL_LAYERS", "1"))
    if layer_count <= 0 or layer_count > base_spec.num_hidden_layers:
        raise ValueError(
            f"QWEN3_PREFILL_LAYERS must be in [1, {base_spec.num_hidden_layers}], got {layer_count}"
        )
    spec = replace(base_spec, num_hidden_layers=layer_count)
    execution_tensors = qwen3_execution_tensors(batch=1, steps=1, spec=spec, max_seq_len=1)
    dispatch_target = DispatchTarget()
    record_qwen3_prefill(dispatch_target, spec=spec, tensors=execution_tensors)
    validate_dispatch_read_write_chain(
        dispatch_target.records,
        initial_tensors=qwen3_prefill_initial_tensors(spec=spec, tensors=execution_tensors),
    ).raise_for_issues()

    unbound = (*qwen3_collect_logical_tensors(execution_tensors), *qwen3_weight_tensors(spec))
    plan = plan_storage(unbound, allocation_id="qwen3-real-prefill")
    bound = bind_storage(unbound, plan)
    tensors = qwen3_tensor_lookup(bound)
    weight_payloads = qwen3_safetensor_weight_payloads(model_dir, spec=spec)

    context = create_compute_context()
    try:
        allocations = allocate_storage_buffers(context, plan)
        resources = qwen3_resource_buffers(context)
        try:
            _write_initial_tensors(
                tensors=tensors,
                allocations=allocations,
                initial=qwen3_prefill_initial_tensors(spec=spec, tensors=execution_tensors),
            )
            write_bound_tensor_payloads(tensors, allocations, weight_payloads)
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
            output_tensor = qwen3_first_tensor(tensors["output.next_token_id"])
            output_bytes = read_bound_tensor_bytes(output_tensor, allocations)
            print(
                "qwen3_safetensor_prefill=ok "
                f"layers={layer_count} dispatches={len(dispatch_target.records)} "
                f"readback_bytes={len(output_bytes)}"
            )
        finally:
            qwen3_close_resource_buffers(resources)
            for allocation in allocations.values():
                allocation.close()
    finally:
        context.close()
    return 0


def _write_initial_tensors(
    *,
    tensors: dict[str, LogicalTensor | tuple[LogicalTensor, ...]],
    allocations: dict[str, VulkanBuffer],
    initial: tuple[LogicalTensor, ...],
) -> None:
    for tensor in initial:
        bound_tensor = qwen3_first_tensor(tensors[tensor.name])
        if tensor.name == "input.input_ids":
            payload = struct.pack("<i", 0)
        elif tensor.name == "input.position_ids":
            payload = struct.pack("<i", 0)
        elif tensor.name == "input.row_indices":
            payload = struct.pack("<q", 0)
        else:
            payload = bytes(tensor_nbytes(bound_tensor))
        write_bound_tensor_bytes(bound_tensor, allocations, payload)


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
