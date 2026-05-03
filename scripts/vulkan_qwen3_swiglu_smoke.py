#!/usr/bin/env python3
"""Run the copied Qwen3 SwiGLU shader through Vulkan."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from torch2vk.logical import activation_tensor
from torch2vk.models.qwen3_safetensor.shaders.swiglu_f32 import SWIGLU_F32
from torch2vk.shader import pack_push_constants
from torch2vk.vulkan_backend import create_compute_context


def main() -> int:
    variant = SWIGLU_F32
    batch = 1
    steps = 1
    width = 4
    gate = (0.0, 1.0, -1.0, 2.0)
    up = (1.0, 2.0, 3.0, 4.0)
    expected = tuple(_silu(a) * b for a, b in zip(gate, up, strict=True))
    tensors = {
        "gate": activation_tensor("gate", dtype="float32", shape=(batch, steps, width)),
        "up": activation_tensor("up", dtype="float32", shape=(batch, steps, width)),
        "output": activation_tensor("output", dtype="float32", shape=(batch, steps, width)),
    }
    symbols = variant.contract.validate(tensors)

    context = create_compute_context()
    try:
        gate_buffer = context.create_host_buffer(nbytes=len(gate) * 4)
        up_buffer = context.create_host_buffer(nbytes=len(up) * 4)
        output_buffer = context.create_host_buffer(nbytes=len(expected) * 4)
        gate_buffer.write(struct.pack(f"<{len(gate)}f", *gate))
        up_buffer.write(struct.pack(f"<{len(up)}f", *up))

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
                    0: gate_buffer,
                    1: up_buffer,
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
            command_buffer.dispatch(1, 1, 1)
            command_buffer.end()
            command_buffer.submit_and_wait(fence)

            actual = struct.unpack(
                f"<{len(expected)}f",
                output_buffer.read(nbytes=len(expected) * 4),
            )
            _assert_close(actual, expected)
            print(f"qwen3_swiglu_dispatch=ok values={actual}")
        finally:
            fence.close()
            command_pool.close()
            pipeline.close()
            pipeline_layout.close()
            descriptor_pool.close()
            descriptor_layout.close()
            module.close()
            output_buffer.close()
            up_buffer.close()
            gate_buffer.close()
    finally:
        context.close()
    return 0


def _silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def _assert_close(actual: tuple[float, ...], expected: tuple[float, ...]) -> None:
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
        if abs(actual_value - expected_value) > 1e-5:
            raise AssertionError(
                f"value mismatch at {index}: actual={actual_value} expected={expected_value}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
