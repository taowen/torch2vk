"""Qwen3 safetensor integration test against a PyTorch reference."""

from __future__ import annotations

import importlib
import os
import struct
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from torch2vk.integration import (
    DebugIntegrationCase,
    first_tensor,
    pytorch_model_reference_provider,
    run_debug_integration_case,
)
from torch2vk.logical import LogicalTensor
from torch2vk.models.qwen3_safetensor.debug import (
    qwen3_prefill_initial_tensors,
)
from torch2vk.models.qwen3_safetensor.execution import (
    run_qwen3_prefill,
)
from torch2vk.models.qwen3_safetensor.runtime import (
    qwen3_close_resource_buffers,
    qwen3_resource_buffers,
)
from torch2vk.models.qwen3_safetensor.spec import load_qwen3_spec
from torch2vk.models.qwen3_safetensor.tensors.prefill import qwen3_prefill_tensors
from torch2vk.models.qwen3_safetensor.tensors.probes import QWEN3_PROBE_TRANSFORMS
from torch2vk.models.qwen3_safetensor.tensors.weights import qwen3_weights
from torch2vk.models.qwen3_safetensor.weights import (
    qwen3_safetensor_weight_payloads,
    verify_qwen3_safetensor_weights,
)
from torch2vk.storage import tensor_nbytes
from torch2vk.vulkan_backend import VulkanBuffer
from torch2vk.vulkan_runner import (
    LogicalTensorLookup,
    write_bound_tensor_bytes,
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
        pytorch_model = _pytorch_qwen3_model(model_dir)

        run_debug_integration_case(
            DebugIntegrationCase(
                shader_dir=SHADER_DIR,
                shader_package=PACKAGE,
                allocation_id="qwen3-integration-prefill",
                tensors=execution_tensors,
                weights=weights,
                initial_tensors=qwen3_prefill_initial_tensors(
                    tensors=execution_tensors,
                    weights=weights,
                ),
                inputs={
                    "input_ids": torch.tensor([PROMPT_IDS], dtype=torch.long),
                    "use_cache": False,
                },
                reference_provider=pytorch_model_reference_provider(pytorch_model),
                run=lambda debug_context: run_qwen3_prefill(
                    debug_context,
                    spec=spec,
                    tensors=execution_tensors,
                    weights=weights,
                ),
                write_initial_tensors=_write_initial_tensors,
                weight_payloads=qwen3_safetensor_weight_payloads(model_dir, spec=spec),
                resource_factory=qwen3_resource_buffers,
                resource_closer=qwen3_close_resource_buffers,
                transforms=QWEN3_PROBE_TRANSFORMS,
                extra_fingerprint={"model_dir": str(model_dir.resolve())},
            )
        )


def _write_initial_tensors(
    tensors: LogicalTensorLookup,
    allocations: Mapping[str, VulkanBuffer],
    initial: tuple[LogicalTensor, ...],
) -> None:
    for tensor in initial:
        bound_tensor = first_tensor(tensors[tensor.name])
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
