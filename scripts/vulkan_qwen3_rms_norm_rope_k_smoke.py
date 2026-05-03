#!/usr/bin/env python3
"""Run the copied Qwen3 fused RMS norm + RoPE K-cache shader through Vulkan."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from torch2vk.logical import activation_tensor, input_tensor, weight_tensor
from torch2vk.models.qwen3_safetensor.shaders.rms_norm_mul_rope_k_f16 import (
    RMS_NORM_MUL_ROPE_K_F16,
)
from torch2vk.shader import pack_push_constants
from torch2vk.vulkan_backend import create_compute_context


def main() -> int:
    variant = RMS_NORM_MUL_ROPE_K_F16
    batch = 1
    steps = 1
    heads = 8
    head_dim = 128
    cache_rows = 4
    cache_stride = heads * head_dim
    target_row = 2
    x = tuple(float((index % 17) + 1) / 16.0 for index in range(batch * steps * heads * head_dim))
    weight = tuple(1.0 + float(index % 7) / 32.0 for index in range(head_dim))
    expected = _rms_norm_reference(x=x, weight=weight, heads=heads, head_dim=head_dim)
    output_elements = cache_rows * cache_stride
    tensors = {
        "x": activation_tensor("x", dtype="float32", shape=(batch, steps, heads, head_dim)),
        "weight": weight_tensor(
            "weight",
            dtype="bfloat16",
            shape=(head_dim,),
            source_key="smoke.weight",
        ),
        "position_ids": input_tensor("position_ids", dtype="int32", shape=(steps,)),
        "freq_factors_placeholder": activation_tensor(
            "freq_factors_placeholder",
            dtype="float32",
            shape=(head_dim,),
        ),
        "row_indices": input_tensor("row_indices", dtype="int64", shape=(steps,)),
        "output": activation_tensor(
            "output",
            dtype="float16",
            shape=(batch, cache_rows, heads, head_dim),
        ),
    }
    symbols = variant.contract.validate(tensors)

    context = create_compute_context()
    try:
        x_buffer = context.create_host_buffer(nbytes=len(x) * 4)
        weight_buffer = context.create_host_buffer(nbytes=len(weight) * 2)
        position_buffer = context.create_host_buffer(nbytes=steps * 4)
        freq_buffer = context.create_host_buffer(nbytes=head_dim * 4)
        output_buffer = context.create_host_buffer(nbytes=output_elements * 2)
        row_indices_buffer = context.create_host_buffer(nbytes=steps * 8)
        x_buffer.write(struct.pack(f"<{len(x)}f", *x))
        weight_buffer.write(_pack_bf16(weight))
        position_buffer.write(struct.pack("<i", 0))
        freq_buffer.write(bytes(head_dim * 4))
        output_buffer.write(bytes(output_elements * 2))
        row_indices_buffer.write(struct.pack("<q", target_row))

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
                    3: position_buffer,
                    4: freq_buffer,
                    5: output_buffer,
                    6: row_indices_buffer,
                },
                descriptor_types={
                    0: "storage_buffer",
                    1: "storage_buffer",
                    3: "storage_buffer",
                    4: "storage_buffer",
                    5: "storage_buffer",
                    6: "storage_buffer",
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
            command_buffer.dispatch(heads, steps, batch)
            command_buffer.end()
            command_buffer.submit_and_wait(fence)

            row_offset = target_row * cache_stride * 2
            actual = struct.unpack(
            f"<{cache_stride}e",
            output_buffer.read(offset=row_offset, nbytes=cache_stride * 2),
        )
            _assert_close(actual, tuple(_f16_round(value) for value in expected))
            untouched_prefix = output_buffer.read(offset=0, nbytes=head_dim * 2)
            if any(untouched_prefix):
                raise AssertionError("K-cache shader wrote before the selected row")
            print(f"qwen3_rms_norm_rope_k_dispatch=ok values={actual[:4]}")
        finally:
            fence.close()
            command_pool.close()
            pipeline.close()
            pipeline_layout.close()
            descriptor_pool.close()
            descriptor_layout.close()
            module.close()
            row_indices_buffer.close()
            output_buffer.close()
            freq_buffer.close()
            position_buffer.close()
            weight_buffer.close()
            x_buffer.close()
    finally:
        context.close()
    return 0


def _rms_norm_reference(
    *,
    x: tuple[float, ...],
    weight: tuple[float, ...],
    heads: int,
    head_dim: int,
) -> tuple[float, ...]:
    rounded_weight = tuple(_bf16_round(value) for value in weight)
    values: list[float] = []
    for head in range(heads):
        row = x[head * head_dim : (head + 1) * head_dim]
        mean = sum(value * value for value in row) / head_dim
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


def _f16_round(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _assert_close(actual: tuple[float, ...], expected: tuple[float, ...]) -> None:
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
        if abs(actual_value - expected_value) > 1e-3:
            raise AssertionError(
                f"value mismatch at {index}: actual={actual_value} expected={expected_value}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
