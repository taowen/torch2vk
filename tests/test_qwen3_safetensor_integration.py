"""Qwen3 safetensor integration test against a PyTorch reference."""

from __future__ import annotations

import importlib
import os
import pkgutil
import struct
import unittest
from pathlib import Path
from typing import Any, cast

import torch

from torch2vk.artifacts import read_bound_tensor_artifacts
from torch2vk.logical import ComparePolicy, LogicalTensor
from torch2vk.models.qwen3_safetensor.debug import (
    qwen3_prefill_debug_boundaries,
    qwen3_prefill_initial_tensors,
)
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
from torch2vk.models.qwen3_safetensor.weights import (
    qwen3_safetensor_weight_payloads,
    verify_qwen3_safetensor_weights,
)
from torch2vk.schema import BoundaryRule
from torch2vk.shader import DispatchTarget, ShaderVariant
from torch2vk.storage import bind_storage, plan_storage, tensor_nbytes
from torch2vk.validation import compare_declared_boundaries, validate_dispatch_read_write_chain
from torch2vk.vulkan_backend import VulkanBuffer, create_compute_context
from torch2vk.vulkan_runner import (
    LogicalTensorLookup,
    VulkanSequenceRunner,
    allocate_storage_buffers,
    write_bound_tensor_bytes,
    write_bound_tensor_payloads,
)

DEFAULT_MODEL_DIR = Path("models/weights/qwen3-0.6b-safetensor")
PACKAGE = "torch2vk.models.qwen3_safetensor.shaders"
SHADER_DIR = Path("build/shaders/qwen3_safetensor")


class Qwen3SafetensorIntegrationTest(unittest.TestCase):
    def test_prefill_prefix_matches_pytorch_reference(self) -> None:
        model_dir = Path(os.environ.get("QWEN3_SAFETENSOR_DIR", str(DEFAULT_MODEL_DIR)))
        if not model_dir.exists():
            self.skipTest(f"missing Qwen3 safetensor model directory: {model_dir}")
        if not SHADER_DIR.exists():
            self.skipTest("Qwen3 shaders are not compiled")

        spec = load_qwen3_spec(model_dir)
        verification = verify_qwen3_safetensor_weights(model_dir, spec=spec)
        verification.raise_for_mismatches()

        execution_tensors = qwen3_execution_tensors(batch=1, steps=1, spec=spec, max_seq_len=1)
        dispatch_target = DispatchTarget()
        record_qwen3_prefill(dispatch_target, spec=spec, tensors=execution_tensors)
        validate_dispatch_read_write_chain(
            dispatch_target.records,
            initial_tensors=qwen3_prefill_initial_tensors(spec=spec, tensors=execution_tensors),
        ).raise_for_issues()

        unbound = (
            *qwen3_collect_logical_tensors(execution_tensors),
            *qwen3_weight_tensors(spec),
        )
        plan = plan_storage(unbound, allocation_id="qwen3-integration-prefill")
        bound = bind_storage(unbound, plan)
        tensors = qwen3_tensor_lookup(bound)
        reference_logits = _pytorch_qwen3_prefill_logits(model_dir)
        reference_token = reference_logits[:, -1].argmax(dim=-1).to(torch.int32)

        context = create_compute_context()
        try:
            allocations = allocate_storage_buffers(context, plan)
            resources = qwen3_resource_buffers(context)
            try:
                _write_initial_tensors(
                    tensors=tensors,
                    allocations=allocations,
                    initial=qwen3_prefill_initial_tensors(
                        spec=spec,
                        tensors=execution_tensors,
                    ),
                )
                write_bound_tensor_payloads(
                    tensors,
                    allocations,
                    qwen3_safetensor_weight_payloads(model_dir, spec=spec),
                )
                runner = VulkanSequenceRunner(
                    context=context,
                    shader_dir=SHADER_DIR,
                    variants=_shader_variants(),
                )
                records = tuple(dispatch_target.records)
                runner.run_bound_storage(
                    records,
                    tensors=tensors,
                    allocations=allocations,
                    resource_buffers=resources,
                )
                report = compare_declared_boundaries(
                    _final_prefill_boundaries(execution_tensors),
                    dispatch_records=records,
                    reference={
                        "output.logits": reference_logits,
                        "output.next_token_id": reference_token,
                    },
                    candidate=read_bound_tensor_artifacts(
                        tensors,
                        allocations,
                        names=("output.logits", "output.next_token_id"),
                    ),
                )
                report.raise_for_mismatch()
            finally:
                qwen3_close_resource_buffers(resources)
                for allocation in allocations.values():
                    allocation.close()
        finally:
            context.close()


def _write_initial_tensors(
    *,
    tensors: LogicalTensorLookup,
    allocations: dict[str, VulkanBuffer],
    initial: tuple[LogicalTensor, ...],
) -> None:
    for tensor in initial:
        bound_tensor = qwen3_first_tensor(tensors[tensor.name])
        if tensor.name in {"input.input_ids", "input.position_ids"}:
            payload = struct.pack("<i", 0)
        elif tensor.name == "input.row_indices":
            payload = struct.pack("<q", 0)
        else:
            payload = bytes(tensor_nbytes(bound_tensor))
        write_bound_tensor_bytes(bound_tensor, allocations, payload)


def _pytorch_qwen3_prefill_logits(model_dir: Path) -> torch.Tensor:
    transformers = cast("Any", importlib.import_module("transformers"))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model.eval()
    with torch.no_grad():
        output = model(input_ids=torch.tensor([[0]], dtype=torch.long), use_cache=False)
    return cast("torch.Tensor", output.logits).float().contiguous()


def _final_prefill_boundaries(tensors: Any) -> tuple[BoundaryRule, ...]:
    debug_boundaries = qwen3_prefill_debug_boundaries(tensors)
    logits = next(boundary for boundary in debug_boundaries if boundary.name == "prefill.logits")
    next_token = next(
        boundary for boundary in debug_boundaries if boundary.name == "prefill.next_token"
    )
    return (
        BoundaryRule(
            name=logits.name,
            phase=logits.phase,
            order=logits.order,
            tensors=logits.tensors,
            compare=ComparePolicy(kind="tensor", rtol=0.0, atol=0.5),
            checkpoint=logits.checkpoint,
            readback=logits.readback,
        ),
        BoundaryRule(
            name=next_token.name,
            phase=next_token.phase,
            order=next_token.order,
            tensors=next_token.tensors,
            compare=ComparePolicy(kind="token"),
            checkpoint=next_token.checkpoint,
            readback=next_token.readback,
        ),
    )


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
