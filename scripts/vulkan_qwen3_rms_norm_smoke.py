#!/usr/bin/env python3
"""Run the Qwen3 RMS norm shader through Vulkan."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from torch2vk.logical import activation_tensor, weight_tensor
from torch2vk.models.qwen3_safetensor.shaders.rms_norm_f32 import RMS_NORM_F32
from torch2vk.shader import pack_push_constants
from torch2vk.vulkan_backend import create_compute_context


def main() -> int:
    variant = RMS_NORM_F32
    batch = 1
    steps = 2
    hidden = 8
    x = tuple(float(index + 1) / 8.0 for index in range(batch * steps * hidden))
    weight = tuple(1.0 + float(index) / 16.0 for index in range(hidden))
    expected = _rms_norm_reference(x=x, weight=weight, steps=steps, hidden=hidden)
    tensors = {
        "x": activation_tensor("x", dtype="float32", shape=(batch, steps, hidden)),
        "weight": weight_tensor(
            "weight",
            dtype="bfloat16",
            shape=(hidden,),
            source_key="smoke.weight",
        ),
        "output": activation_tensor("output", dtype="float32", shape=(batch, steps, hidden)),
    }
    symbols = variant.contract.validate(tensors)

    context = create_compute_context()
    try:
        x_buffer = context.create_host_buffer(nbytes=len(x) * 4)
        weight_buffer = context.create_host_buffer(nbytes=len(weight) * 2)
        output_buffer = context.create_host_buffer(nbytes=len(expected) * 4)
        x_buffer.write(struct.pack(f"<{len(x)}f", *x))
        weight_buffer.write(_pack_bf16(weight))

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
                    0: x_buffer,
                    1: weight_buffer,
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
            command_buffer.dispatch(steps, batch, 1)
            command_buffer.end()
            command_buffer.submit_and_wait(fence)

            actual = struct.unpack(
                f"<{len(expected)}f",
                output_buffer.read(nbytes=len(expected) * 4),
            )
            _assert_close(actual, expected)
            print(f"qwen3_rms_norm_dispatch=ok values={actual[:4]}")
        finally:
            fence.close()
            command_pool.close()
            pipeline.close()
            pipeline_layout.close()
            descriptor_pool.close()
            descriptor_layout.close()
            module.close()
            output_buffer.close()
            weight_buffer.close()
            x_buffer.close()
    finally:
        context.close()
    return 0


def _rms_norm_reference(
    *,
    x: tuple[float, ...],
    weight: tuple[float, ...],
    steps: int,
    hidden: int,
) -> tuple[float, ...]:
    rounded_weight = tuple(_bf16_round(value) for value in weight)
    values: list[float] = []
    for step in range(steps):
        row = x[step * hidden : (step + 1) * hidden]
        mean = sum(value * value for value in row) / hidden
        scale = 1.0 / math.sqrt(mean + 1.0e-6)
        values.extend(value * scale * rounded_weight[index] for index, value in enumerate(row))
    return tuple(values)


def _pack_bf16(values: tuple[float, ...]) -> bytes:
    packed = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        packed.extend(struct.pack("<H", bits >> 16))
    return bytes(packed)


def _bf16_round(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", (bits >> 16) << 16))[0]


def _assert_close(actual: tuple[float, ...], expected: tuple[float, ...]) -> None:
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
        if abs(actual_value - expected_value) > 1e-5:
            raise AssertionError(
                f"value mismatch at {index}: actual={actual_value} expected={expected_value}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
