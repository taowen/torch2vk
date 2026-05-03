#!/usr/bin/env python3
"""Run the Qwen3 two-stage argmax shaders through Vulkan."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from torch2vk.logical import activation_tensor, output_tensor
from torch2vk.models.qwen3_safetensor.shaders.argmax_last_logits_f32_stage1 import (
    ARGMAX_LAST_LOGITS_STAGE1,
)
from torch2vk.models.qwen3_safetensor.shaders.argmax_last_logits_f32_stage2 import (
    ARGMAX_LAST_LOGITS_STAGE2,
)
from torch2vk.shader import ShaderVariant, pack_uniform_blocks
from torch2vk.vulkan_backend import VulkanBuffer, create_compute_context


def main() -> int:
    batch = 2
    steps = 2
    vocab = 1300
    chunks = (vocab + 1023) // 1024
    logits = [0.0 for _ in range(batch * steps * vocab)]
    _set_last_step_value(logits, batch=0, steps=steps, vocab=vocab, token=7, value=3.0)
    _set_last_step_value(logits, batch=0, steps=steps, vocab=vocab, token=1200, value=9.0)
    _set_last_step_value(logits, batch=1, steps=steps, vocab=vocab, token=5, value=4.0)
    _set_last_step_value(logits, batch=1, steps=steps, vocab=vocab, token=1025, value=4.0)
    expected = (1200, 5)

    context = create_compute_context()
    try:
        logits_buffer = context.create_host_buffer(nbytes=len(logits) * 4)
        partial_values = context.create_host_buffer(nbytes=batch * chunks * 4)
        partial_indices = context.create_host_buffer(nbytes=batch * chunks * 4)
        output = context.create_host_buffer(nbytes=batch * 4)
        logits_buffer.write(struct.pack(f"<{len(logits)}f", *logits))

        stage1_uniform = _uniform_buffer(
            context,
            ARGMAX_LAST_LOGITS_STAGE1,
            {
                "logits": activation_tensor("logits", dtype="float32", shape=(batch, steps, vocab)),
                "partial_values": activation_tensor(
                    "partial_values", dtype="float32", shape=(batch, chunks)
                ),
                "partial_indices": activation_tensor(
                    "partial_indices", dtype="int32", shape=(batch, chunks)
                ),
            },
        )
        stage2_uniform = _uniform_buffer(
            context,
            ARGMAX_LAST_LOGITS_STAGE2,
            {
                "partial_values": activation_tensor(
                    "partial_values", dtype="float32", shape=(batch, chunks)
                ),
                "partial_indices": activation_tensor(
                    "partial_indices", dtype="int32", shape=(batch, chunks)
                ),
                "output": output_tensor("output", dtype="int32", shape=(batch,)),
            },
        )
        try:
            _run_variant(
                context,
                ARGMAX_LAST_LOGITS_STAGE1,
                descriptors={
                    0: logits_buffer,
                    1: partial_values,
                    2: partial_indices,
                    3: stage1_uniform,
                },
                descriptor_types={
                    0: "storage_buffer",
                    1: "storage_buffer",
                    2: "storage_buffer",
                    3: "uniform_buffer",
                },
                dispatch=(chunks, batch, 1),
            )
            _run_variant(
                context,
                ARGMAX_LAST_LOGITS_STAGE2,
                descriptors={
                    0: partial_values,
                    1: partial_indices,
                    2: output,
                    3: stage2_uniform,
                },
                descriptor_types={
                    0: "storage_buffer",
                    1: "storage_buffer",
                    2: "storage_buffer",
                    3: "uniform_buffer",
                },
                dispatch=(1, 1, batch),
            )
            actual = struct.unpack(f"<{batch}i", output.read(nbytes=batch * 4))
            if actual != expected:
                raise AssertionError(f"argmax mismatch actual={actual} expected={expected}")
            print(f"qwen3_argmax_dispatch=ok tokens={actual}")
        finally:
            stage2_uniform.close()
            stage1_uniform.close()
            output.close()
            partial_indices.close()
            partial_values.close()
            logits_buffer.close()
    finally:
        context.close()
    return 0


def _set_last_step_value(
    logits: list[float],
    *,
    batch: int,
    steps: int,
    vocab: int,
    token: int,
    value: float,
) -> None:
    logits[((batch * steps) + (steps - 1)) * vocab + token] = value


def _uniform_buffer(context, variant: ShaderVariant, tensors) -> VulkanBuffer:
    symbols = variant.contract.validate(tensors)
    uniform = context.create_host_buffer(nbytes=16)
    uniform.write(pack_uniform_blocks(variant.contract, symbols)["sizes"])
    return uniform


def _run_variant(
    context,
    variant: ShaderVariant,
    *,
    descriptors: dict[int, VulkanBuffer],
    descriptor_types: dict[int, str],
    dispatch: tuple[int, int, int],
) -> None:
    module = context.create_shader_module(
        Path(f"build/shaders/qwen3_safetensor/{variant.name}.spv").read_bytes()
    )
    descriptor_layout = context.create_descriptor_set_layout(variant.contract)
    descriptor_pool = context.create_descriptor_pool(variant.contract)
    pipeline_layout = context.create_pipeline_layout(variant.contract, descriptor_layout)
    pipeline = context.create_compute_pipeline(shader_module=module, pipeline_layout=pipeline_layout)
    command_pool = context.create_command_pool()
    fence = context.create_fence()
    try:
        descriptor_set = context.allocate_descriptor_set(
            descriptor_pool=descriptor_pool,
            descriptor_set_layout=descriptor_layout,
        )
        context.update_descriptor_set(
            descriptor_set,
            descriptors,
            descriptor_types=descriptor_types,
        )
        command_buffer = command_pool.allocate_command_buffer()
        command_buffer.begin()
        command_buffer.bind_compute_pipeline(pipeline)
        command_buffer.bind_descriptor_set(
            pipeline_layout=pipeline_layout,
            descriptor_set=descriptor_set,
        )
        command_buffer.dispatch(*dispatch)
        command_buffer.end()
        command_buffer.submit_and_wait(fence)
    finally:
        fence.close()
        command_pool.close()
        pipeline.close()
        pipeline_layout.close()
        descriptor_pool.close()
        descriptor_layout.close()
        module.close()


if __name__ == "__main__":
    raise SystemExit(main())
