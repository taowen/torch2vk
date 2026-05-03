#!/usr/bin/env python3
"""Run the copied Qwen3 BF16 embedding shader through Vulkan."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from torch2vk.logical import input_tensor, output_tensor, weight_tensor
from torch2vk.models.qwen3_safetensor.shaders.embedding_lookup_bf16_f32_sequence import (
    EMBEDDING_LOOKUP_BF16_F32,
)
from torch2vk.shader import pack_uniform_blocks
from torch2vk.vulkan_backend import create_compute_context


def main() -> int:
    variant = EMBEDDING_LOOKUP_BF16_F32
    batch = 1
    steps = 2
    hidden = 4
    vocab = 4
    input_ids = (2, 1)
    weights = tuple(float(index) / 4.0 for index in range(vocab * hidden))
    expected = _embedding_reference(
        weights=weights,
        input_ids=input_ids,
        steps=steps,
        hidden=hidden,
    )
    tensors = {
        "input_ids": input_tensor("input_ids", dtype="int32", shape=(batch, steps)),
        "weight": weight_tensor(
            "weight",
            dtype="bfloat16",
            shape=(vocab, hidden),
            source_key="smoke.weight",
        ),
        "output": output_tensor("output", dtype="float32", shape=(batch, steps, hidden)),
    }
    symbols = variant.contract.validate(tensors)

    context = create_compute_context()
    try:
        output = context.create_host_buffer(nbytes=batch * steps * hidden * 4)
        ids = context.create_host_buffer(nbytes=len(input_ids) * 4)
        weight = context.create_host_buffer(nbytes=len(weights) * 2)
        sizes = context.create_host_buffer(nbytes=16)
        ids.write(struct.pack(f"<{len(input_ids)}i", *input_ids))
        weight.write(_pack_bf16(weights))
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
            command_buffer.dispatch(math.ceil(hidden / 512), steps, batch)
            command_buffer.end()
            command_buffer.submit_and_wait(fence)

            actual = struct.unpack(f"<{len(expected)}f", output.read(nbytes=len(expected) * 4))
            _assert_close(actual, expected)
            print(f"qwen3_embedding_dispatch=ok values={actual}")
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


def _embedding_reference(
    *,
    weights: tuple[float, ...],
    input_ids: tuple[int, ...],
    steps: int,
    hidden: int,
) -> tuple[float, ...]:
    values: list[float] = []
    for step in range(steps):
        token = input_ids[step]
        row = weights[token * hidden : (token + 1) * hidden]
        values.extend(_bf16_round(value) for value in row)
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
    if len(actual) != len(expected):
        raise AssertionError(f"length mismatch actual={len(actual)} expected={len(expected)}")
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
        if abs(actual_value - expected_value) > 1e-6:
            raise AssertionError(
                f"value mismatch at {index}: actual={actual_value} expected={expected_value}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
