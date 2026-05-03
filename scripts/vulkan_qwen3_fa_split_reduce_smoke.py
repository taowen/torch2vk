#!/usr/bin/env python3
"""Run the Qwen3 flash-attention split-k reduce shader through Vulkan."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from torch2vk.logical import activation_tensor
from torch2vk.models.qwen3_safetensor.shaders.fa_split_k_reduce import FA_SPLIT_K_REDUCE
from torch2vk.shader import pack_push_constants
from torch2vk.vulkan_backend import create_compute_context


def main() -> int:
    variant = FA_SPLIT_K_REDUCE
    batch = 1
    steps = 1
    q_heads = 1
    head_dim = 4
    k_num = 4
    split_values = (
        1.0,
        2.0,
        3.0,
        4.0,
        2.0,
        3.0,
        4.0,
        5.0,
        3.0,
        4.0,
        5.0,
        6.0,
        4.0,
        5.0,
        6.0,
        7.0,
    )
    split_k_input = _split_k_buffer(
        values=split_values,
        head_dim=head_dim,
        q_heads=q_heads,
        steps=steps,
        batch=batch,
        k_num=k_num,
    )
    expected = (2.5, 3.5, 4.5, 5.5)
    tensors = {
        "split_k_input": activation_tensor(
            "split_k_input",
            dtype="float32",
            shape=(len(split_k_input),),
        ),
        "sinks_placeholder": activation_tensor(
            "sinks_placeholder",
            dtype="float32",
            shape=(batch, steps, q_heads, head_dim),
        ),
        "output": activation_tensor(
            "output",
            dtype="float32",
            shape=(batch, steps, q_heads, head_dim),
        ),
    }
    symbols = variant.contract.validate(tensors)

    context = create_compute_context()
    try:
        split_buffer = context.create_host_buffer(nbytes=len(split_k_input) * 4)
        sinks_buffer = context.create_host_buffer(nbytes=head_dim * q_heads * steps * batch * 4)
        output_buffer = context.create_host_buffer(nbytes=len(expected) * 4)
        split_buffer.write(struct.pack(f"<{len(split_k_input)}f", *split_k_input))
        sinks_buffer.write(bytes(head_dim * q_heads * steps * batch * 4))

        module = context.create_shader_module(
            Path(f"build/shaders/qwen3_safetensor/{variant.name}.spv").read_bytes()
        )
        descriptor_layout = context.create_descriptor_set_layout(variant.contract)
        descriptor_pool = context.create_descriptor_pool(variant.contract)
        pipeline_layout = context.create_pipeline_layout(variant.contract, descriptor_layout)
        pipeline = context.create_compute_pipeline(
            shader_module=module,
            pipeline_layout=pipeline_layout,
            specialization_constants=variant.specialization_constants,
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
                    0: split_buffer,
                    1: sinks_buffer,
                    2: output_buffer,
                },
                descriptor_types={
                    0: "storage_buffer",
                    1: "storage_buffer",
                    2: "storage_buffer",
                },
            )
            command_buffer = command_pool.allocate_command_buffer()
            command_buffer.begin()
            command_buffer.bind_compute_pipeline(pipeline)
            command_buffer.bind_descriptor_set(
                pipeline_layout=pipeline_layout,
                descriptor_set=descriptor_set,
            )
            command_buffer.push_constants(
                pipeline_layout=pipeline_layout,
                data=pack_push_constants(variant.contract, tensors, symbols) or b"",
            )
            command_buffer.dispatch(q_heads, math.ceil(head_dim / 64), batch * steps)
            command_buffer.end()
            command_buffer.submit_and_wait(fence)

            actual = struct.unpack(
                f"<{len(expected)}f",
                output_buffer.read(nbytes=len(expected) * 4),
            )
            _assert_close(actual, expected)
            print(f"qwen3_fa_split_reduce_dispatch=ok values={actual}")
        finally:
            fence.close()
            command_pool.close()
            pipeline.close()
            pipeline_layout.close()
            descriptor_pool.close()
            descriptor_layout.close()
            module.close()
            output_buffer.close()
            sinks_buffer.close()
            split_buffer.close()
    finally:
        context.close()
    return 0


def _split_k_buffer(
    *,
    values: tuple[float, ...],
    head_dim: int,
    q_heads: int,
    steps: int,
    batch: int,
    k_num: int,
) -> tuple[float, ...]:
    output_count = head_dim * q_heads * steps * batch * k_num
    lm_count = q_heads * 2 * k_num * steps * batch
    data = [0.0 for _ in range(output_count + lm_count)]
    for split in range(k_num):
        for dim in range(head_dim):
            data[head_dim * q_heads * split + dim] = values[split * head_dim + dim]
        lm_base = output_count + q_heads * 2 * split
        data[lm_base] = 1.0
        data[lm_base + q_heads] = 0.0
    return tuple(data)


def _assert_close(actual: tuple[float, ...], expected: tuple[float, ...]) -> None:
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
        if abs(actual_value - expected_value) > 1e-5:
            raise AssertionError(
                f"value mismatch at {index}: actual={actual_value} expected={expected_value}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
