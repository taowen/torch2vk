"""Qwen3 safetensor integration test against a PyTorch reference."""

from __future__ import annotations

import importlib
import os
import pkgutil
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import torch

from torch2vk.debug_context import DebugContext
from torch2vk.logical import LogicalTensor
from torch2vk.models.qwen3_safetensor.debug import (
    qwen3_prefill_initial_tensors,
)
from torch2vk.models.qwen3_safetensor.execution import (
    run_qwen3_prefill,
)
from torch2vk.models.qwen3_safetensor.runtime import (
    qwen3_close_resource_buffers,
    qwen3_collect_logical_tensors,
    qwen3_first_tensor,
    qwen3_resource_buffers,
    qwen3_tensor_lookup,
)
from torch2vk.models.qwen3_safetensor.spec import load_qwen3_spec
from torch2vk.models.qwen3_safetensor.tensors.prefill import qwen3_prefill_tensors
from torch2vk.models.qwen3_safetensor.tensors.probes import QWEN3_PROBE_TRANSFORMS
from torch2vk.models.qwen3_safetensor.tensors.weights import qwen3_weights
from torch2vk.models.qwen3_safetensor.weights import (
    qwen3_safetensor_weight_payloads,
    verify_qwen3_safetensor_weights,
)
from torch2vk.pytorch import ArtifactCache
from torch2vk.shader import ShaderVariant
from torch2vk.storage import bind_storage, plan_storage, tensor_nbytes
from torch2vk.validation import validate_dispatch_read_write_chain
from torch2vk.vulkan_backend import VulkanBuffer, create_compute_context
from torch2vk.vulkan_runner import (
    LogicalTensorLookup,
    allocate_storage_buffers,
    write_bound_tensor_bytes,
    write_bound_tensor_payloads,
)

DEFAULT_MODEL_DIR = Path("models/weights/qwen3-0.6b-safetensor")
PACKAGE = "torch2vk.models.qwen3_safetensor.shaders"
SHADER_DIR = Path("build/shaders/qwen3_safetensor")
PROMPT_IDS = (0, 1)


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

        execution_tensors = qwen3_prefill_tensors(
            batch=1,
            steps=len(PROMPT_IDS),
            spec=spec,
            max_seq_len=len(PROMPT_IDS),
        )
        weights = qwen3_weights(spec)
        unbound = (
            *qwen3_collect_logical_tensors(execution_tensors),
            *qwen3_collect_logical_tensors(weights),
        )
        plan = plan_storage(unbound, allocation_id="qwen3-integration-prefill")
        bound = bind_storage(unbound, plan)
        tensors = qwen3_tensor_lookup(bound)
        pytorch_model = _pytorch_qwen3_model(model_dir)

        context = create_compute_context()
        try:
            allocations = allocate_storage_buffers(context, plan)
            resources = qwen3_resource_buffers(context)
            try:
                _write_initial_tensors(
                    tensors=tensors,
                    allocations=allocations,
                    initial=qwen3_prefill_initial_tensors(
                        tensors=execution_tensors,
                        weights=weights,
                    ),
                )
                write_bound_tensor_payloads(
                    tensors,
                    allocations,
                    qwen3_safetensor_weight_payloads(model_dir, spec=spec),
                )
                with tempfile.TemporaryDirectory() as cache_dir:
                    debug_context = DebugContext(
                        shader_dir=SHADER_DIR,
                        variants=_shader_variants(),
                        context=context,
                        tensors=tensors,
                        tensor_sequence=bound,
                        allocations=allocations,
                        inputs={
                            "input_ids": torch.tensor([PROMPT_IDS], dtype=torch.long),
                            "use_cache": False,
                        },
                        cache=ArtifactCache(Path(cache_dir)),
                        resource_buffers=resources,
                        transforms=QWEN3_PROBE_TRANSFORMS,
                        extra_fingerprint={"model_dir": str(model_dir.resolve())},
                    )
                    run_qwen3_prefill(
                        debug_context,
                        pytorch_model,
                        spec=spec,
                        tensors=execution_tensors,
                        weights=weights,
                    )
                    validate_dispatch_read_write_chain(
                        debug_context.records,
                        initial_tensors=qwen3_prefill_initial_tensors(
                            tensors=execution_tensors,
                            weights=weights,
                        ),
                    ).raise_for_issues()
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
            values = (
                PROMPT_IDS
                if tensor.name == "input.input_ids"
                else tuple(range(len(PROMPT_IDS)))
            )
            payload = struct.pack(f"<{len(PROMPT_IDS)}i", *values)
        elif tensor.name == "input.row_indices":
            payload = struct.pack(f"<{len(PROMPT_IDS)}q", *range(len(PROMPT_IDS)))
        elif tensor.name == "input.attention_mask_f16":
            mask = tuple(
                0.0 if key <= query else -65504.0
                for query in range(len(PROMPT_IDS))
                for key in range(len(PROMPT_IDS))
            )
            payload = struct.pack(f"<{len(mask)}e", *mask)
        else:
            payload = bytes(tensor_nbytes(bound_tensor))
        write_bound_tensor_bytes(bound_tensor, allocations, payload)


def _pytorch_qwen3_model(model_dir: Path) -> Any:
    transformers = cast("Any", importlib.import_module("transformers"))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model.eval()
    return model


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
