#!/usr/bin/env python3
"""Run the copied Qwen3 set-rows KV-cache write shader through Vulkan."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from torch2vk.logical import activation_tensor, input_tensor
from torch2vk.models.qwen3_safetensor.shaders.set_rows_f16_i64_token_major import (
    SET_ROWS_F16_I64_TOKEN_MAJOR,
)
from torch2vk.shader import pack_push_constants
from torch2vk.vulkan_backend import create_compute_context


def main() -> int:
    variant = SET_ROWS_F16_I64_TOKEN_MAJOR
    batch = 1
    steps = 2
    cache_steps = 4
    width = 4
    x = tuple(float(index + 1) / 8.0 for index in range(batch * steps * width))
    row_indices = (2, 0)
    expected = _set_rows_reference(
        x=x,
        row_indices=row_indices,
        steps=steps,
        cache_steps=cache_steps,
        width=width,
    )
    tensors = {
        "x": activation_tensor("x", dtype="float32", shape=(1, batch, steps, width)),
        "row_indices": input_tensor("row_indices", dtype="int64", shape=(steps,)),
        "output": activation_tensor(
            "output",
            dtype="float16",
            shape=(1, batch, cache_steps, width),
        ),
    }
    symbols = variant.contract.validate(tensors)

    context = create_compute_context()
    try:
        x_buffer = context.create_host_buffer(nbytes=len(x) * 4)
        row_indices_buffer = context.create_host_buffer(nbytes=len(row_indices) * 8)
        output_buffer = context.create_host_buffer(nbytes=len(expected) * 2)
        x_buffer.write(struct.pack(f"<{len(x)}f", *x))
        row_indices_buffer.write(struct.pack(f"<{len(row_indices)}q", *row_indices))
        output_buffer.write(bytes(len(expected) * 2))

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
                {
                    0: x_buffer,
                    1: row_indices_buffer,
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
            command_buffer.dispatch(math.ceil(len(x) / 512), 1, 1)
            command_buffer.end()
            command_buffer.submit_and_wait(fence)

            actual = struct.unpack(
                f"<{len(expected)}e",
                output_buffer.read(nbytes=len(expected) * 2),
            )
            _assert_close(actual, expected)
            print(f"qwen3_set_rows_dispatch=ok values={actual}")
        finally:
            fence.close()
            command_pool.close()
            pipeline.close()
            pipeline_layout.close()
            descriptor_pool.close()
            descriptor_layout.close()
            module.close()
            output_buffer.close()
            row_indices_buffer.close()
            x_buffer.close()
    finally:
        context.close()
    return 0


def _set_rows_reference(
    *,
    x: tuple[float, ...],
    row_indices: tuple[int, ...],
    steps: int,
    cache_steps: int,
    width: int,
) -> tuple[float, ...]:
    values = [0.0 for _ in range(cache_steps * width)]
    for step in range(steps):
        row = row_indices[step]
        for col in range(width):
            values[row * width + col] = x[step * width + col]
    return tuple(values)


def _assert_close(actual: tuple[float, ...], expected: tuple[float, ...]) -> None:
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
        if abs(actual_value - expected_value) > 1e-3:
            raise AssertionError(
                f"value mismatch at {index}: actual={actual_value} expected={expected_value}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
