#!/usr/bin/env python3
"""Run Qwen3 embedding shader against real safetensor BF16 weights."""

from __future__ import annotations

import struct
from pathlib import Path

import torch
from safetensors import safe_open

from torch2vk.logical import input_tensor, output_tensor, weight_tensor
from torch2vk.models.qwen3_safetensor.shaders.embedding_lookup_bf16_f32_sequence import (
    EMBEDDING_LOOKUP_BF16_F32,
)
from torch2vk.models.qwen3_safetensor.spec import load_qwen3_spec
from torch2vk.models.qwen3_safetensor.weights import verify_qwen3_safetensor_weights
from torch2vk.shader import pack_uniform_blocks
from torch2vk.vulkan_backend import create_compute_context


MODEL_DIR = Path("models/weights/qwen3-0.6b-safetensor")
WEIGHT_KEY = "model.embed_tokens.weight"


def main() -> int:
    spec = load_qwen3_spec(MODEL_DIR)
    verification = verify_qwen3_safetensor_weights(MODEL_DIR, spec=spec)
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
    symbols = variant.contract.validate(tensors)

    context = create_compute_context()
    try:
        output = context.create_host_buffer(nbytes=expected.numel() * 4)
        ids = context.create_host_buffer(nbytes=len(input_ids) * 4)
        weight = context.create_host_buffer(nbytes=embed.numel() * 2)
        sizes = context.create_host_buffer(nbytes=16)
        ids.write(struct.pack(f"<{len(input_ids)}i", *input_ids))
        weight.write(embed.view(torch.uint16).numpy().tobytes())
        sizes.write(pack_uniform_blocks(variant.contract, symbols)["sizes"])

        module = context.create_shader_module(
            Path(f"build/shaders/qwen3_safetensor/{variant.name}.spv").read_bytes()
        )
        descriptor_layout = context.create_descriptor_set_layout(variant.contract)
        descriptor_pool = context.create_descriptor_pool(variant.contract)
        pipeline_layout = context.create_pipeline_layout(variant.contract, descriptor_layout)
        pipeline = context.create_compute_pipeline(
            shader_module=module,
            pipeline_layout=pipeline_layout,
        )
        command_pool = context.create_command_pool()
        fence = context.create_fence()
        try:
            descriptor_set = context.allocate_descriptor_set(
                descriptor_pool=descriptor_pool,
                descriptor_set_layout=descriptor_layout,
            )
            context.update_descriptor_set(
                descriptor_set,
                {
                    0: output,
                    1: ids,
                    2: weight,
                    3: sizes,
                },
                descriptor_types={
                    0: "storage_buffer",
                    1: "storage_buffer",
                    2: "storage_buffer",
                    3: "uniform_buffer",
                },
            )
            command_buffer = command_pool.allocate_command_buffer()
            command_buffer.begin()
            command_buffer.bind_compute_pipeline(pipeline)
            command_buffer.bind_descriptor_set(
                pipeline_layout=pipeline_layout,
                descriptor_set=descriptor_set,
            )
            command_buffer.dispatch((spec.hidden_size + 511) // 512, steps, batch)
            command_buffer.end()
            command_buffer.submit_and_wait(fence)

            actual = torch.frombuffer(
                bytearray(output.read(nbytes=expected.numel() * 4)),
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
            fence.close()
            command_pool.close()
            pipeline.close()
            pipeline_layout.close()
            descriptor_pool.close()
            descriptor_layout.close()
            module.close()
            sizes.close()
            weight.close()
            ids.close()
            output.close()
    finally:
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
