"""Qwen3 safetensor integration test against a PyTorch reference."""

from __future__ import annotations

import os
import struct
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from safetensors import safe_open

from torch2vk.artifacts import read_bound_tensor_artifacts
from torch2vk.logical import ComparePolicy, input_tensor, output_tensor, weight_tensor
from torch2vk.models.qwen3_safetensor.shaders.embedding_lookup_bf16_f32_sequence import (
    EMBEDDING_LOOKUP_BF16_F32,
)
from torch2vk.models.qwen3_safetensor.spec import load_qwen3_spec
from torch2vk.models.qwen3_safetensor.weights import (
    qwen3_safetensor_weight_bytes,
    verify_qwen3_safetensor_weights,
)
from torch2vk.schema import BoundaryRule
from torch2vk.shader import DispatchTarget
from torch2vk.storage import bind_storage, plan_storage
from torch2vk.validation import compare_declared_boundaries
from torch2vk.vulkan_backend import create_compute_context
from torch2vk.vulkan_runner import (
    VulkanSequenceRunner,
    allocate_storage_buffers,
    write_bound_tensor_bytes,
    write_bound_tensor_payloads,
)

if TYPE_CHECKING:
    import torch

DEFAULT_MODEL_DIR = Path("models/weights/qwen3-0.6b-safetensor")
SHADER_DIR = Path("build/shaders/qwen3_safetensor")
WEIGHT_KEY = "model.embed_tokens.weight"


class Qwen3SafetensorIntegrationTest(unittest.TestCase):
    def test_embedding_shader_matches_pytorch_reference(self) -> None:
        model_dir = Path(os.environ.get("QWEN3_SAFETENSOR_DIR", str(DEFAULT_MODEL_DIR)))
        if not model_dir.exists():
            self.skipTest(f"missing Qwen3 safetensor model directory: {model_dir}")
        if not (SHADER_DIR / f"{EMBEDDING_LOOKUP_BF16_F32.name}.spv").exists():
            self.skipTest("Qwen3 shaders are not compiled")

        spec = load_qwen3_spec(model_dir)
        verification = verify_qwen3_safetensor_weights(model_dir, spec=spec)
        verification.raise_for_mismatches()
        checkpoint_tensor = verification.checkpoint_tensors[WEIGHT_KEY]
        safetensor_handle = cast(
            "Any",
            safe_open(checkpoint_tensor.shard, framework="pt", device="cpu"),
        )
        with safetensor_handle as handle:
            embed = cast("torch.Tensor", handle.get_tensor(WEIGHT_KEY)).contiguous()

        batch = 1
        steps = 2
        input_ids = (0, 1)
        expected = (
            embed[list(input_ids)]
            .float()
            .reshape(batch, steps, spec.hidden_size)
            .contiguous()
        )
        variant = EMBEDDING_LOOKUP_BF16_F32
        tensors = {
            "input_ids": input_tensor("input_ids", dtype="int32", shape=(batch, steps)),
            "weight": weight_tensor(
                "weight",
                dtype="bfloat16",
                shape=(spec.vocab_size, spec.hidden_size),
                source_key=WEIGHT_KEY,
            ),
            "output": output_tensor(
                "output",
                dtype="float32",
                shape=(batch, steps, spec.hidden_size),
            ),
        }
        target = DispatchTarget()
        variant(target, **tensors)
        unbound = tuple(tensors.values())
        plan = plan_storage(unbound, allocation_id="qwen3-integration-embedding")
        bound = {tensor.name: tensor for tensor in bind_storage(unbound, plan)}

        context = create_compute_context()
        try:
            allocations = allocate_storage_buffers(context, plan)
            try:
                write_bound_tensor_bytes(
                    bound["input_ids"],
                    allocations,
                    struct.pack(f"<{len(input_ids)}i", *input_ids),
                )
                write_bound_tensor_payloads(
                    bound,
                    allocations,
                    {
                        "weight": qwen3_safetensor_weight_bytes(
                            model_dir,
                            tensors["weight"],
                            spec=spec,
                        )
                    },
                )
                runner = VulkanSequenceRunner(
                    context=context,
                    shader_dir=SHADER_DIR,
                    variants={variant.name: variant},
                )
                runner.run_bound_storage(
                    tuple(target.records),
                    tensors=bound,
                    allocations=allocations,
                )
                report = compare_declared_boundaries(
                    (
                        BoundaryRule(
                            name="embedding",
                            phase="model",
                            order=0,
                            tensors=(bound["output"],),
                            compare=ComparePolicy(kind="tensor", rtol=0.0, atol=0.0),
                            checkpoint=bound["input_ids"],
                            readback="writer-io",
                        ),
                    ),
                    dispatch_records=tuple(target.records),
                    reference={"output": expected},
                    candidate=read_bound_tensor_artifacts(bound, allocations, names=("output",)),
                )
                report.raise_for_mismatch()
            finally:
                for allocation in allocations.values():
                    allocation.close()
        finally:
            context.close()
