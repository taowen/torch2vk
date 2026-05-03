#!/usr/bin/env python3
"""Run Qwen3 embedding shader against real safetensor BF16 weights."""

from __future__ import annotations

import struct
import os
from pathlib import Path

import torch
from safetensors import safe_open

from torch2vk.logical import input_tensor, output_tensor, weight_tensor
from torch2vk.models.qwen3_safetensor.shaders.embedding_lookup_bf16_f32_sequence import (
    EMBEDDING_LOOKUP_BF16_F32,
)
from torch2vk.models.qwen3_safetensor.spec import load_qwen3_spec
from torch2vk.models.qwen3_safetensor.weights import (
    qwen3_safetensor_weight_bytes,
    verify_qwen3_safetensor_weights,
)
from torch2vk.shader import DispatchTarget
from torch2vk.storage import bind_storage, plan_storage
from torch2vk.vulkan_backend import create_compute_context
from torch2vk.vulkan_runner import (
    VulkanSequenceRunner,
    allocate_storage_buffers,
    read_bound_tensor_bytes,
    write_bound_tensor_bytes,
)


DEFAULT_MODEL_DIR = Path("models/weights/qwen3-0.6b-safetensor")
WEIGHT_KEY = "model.embed_tokens.weight"


def main() -> int:
    model_dir = Path(os.environ.get("QWEN3_SAFETENSOR_DIR", str(DEFAULT_MODEL_DIR)))
    if not model_dir.exists():
        print(f"qwen3_safetensor_embedding_dispatch=skip reason=missing_model_dir path={model_dir}")
        return 0
    spec = load_qwen3_spec(model_dir)
    verification = verify_qwen3_safetensor_weights(model_dir, spec=spec)
    verification.raise_for_mismatches()
    checkpoint_tensor = verification.checkpoint_tensors[WEIGHT_KEY]
    with safe_open(checkpoint_tensor.shard, framework="pt", device="cpu") as handle:
        embed = handle.get_tensor(WEIGHT_KEY).contiguous()

    batch = 1
    steps = 2
    input_ids = (0, 1)
    expected = embed[list(input_ids)].float().reshape(batch, steps, spec.hidden_size).contiguous()
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
    plan = plan_storage(unbound, allocation_id="qwen3-real-embedding")
    bound = {tensor.name: tensor for tensor in bind_storage(unbound, plan)}

    context = create_compute_context()
    try:
        allocations = allocate_storage_buffers(context, plan)
        write_bound_tensor_bytes(
            bound["input_ids"],
            allocations,
            struct.pack(f"<{len(input_ids)}i", *input_ids),
        )
        write_bound_tensor_bytes(
            bound["weight"],
            allocations,
            qwen3_safetensor_weight_bytes(model_dir, tensors["weight"], spec=spec),
        )
        runner = VulkanSequenceRunner(
            context=context,
            shader_dir=Path("build/shaders/qwen3_safetensor"),
            variants={variant.name: variant},
        )
        try:
            runner.run_bound_storage(
                tuple(target.records),
                tensors=bound,
                allocations=allocations,
            )
            actual = torch.frombuffer(
                bytearray(read_bound_tensor_bytes(bound["output"], allocations)),
                dtype=torch.float32,
            ).reshape_as(expected)
            if not actual.allclose(expected, rtol=0.0, atol=0.0):
                max_abs = (actual - expected).abs().max().item()
                raise AssertionError(f"embedding mismatch max_abs={max_abs}")
            print(
                "qwen3_safetensor_embedding_dispatch=ok "
                f"tokens={input_ids} hidden={spec.hidden_size}"
            )
        finally:
            for allocation in allocations.values():
                allocation.close()
    finally:
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
