#!/usr/bin/env python3
"""Run Qwen3 shaders through the reusable Vulkan dispatch runner."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from torch2vk.logical import activation_tensor, input_tensor, output_tensor, weight_tensor
from torch2vk.models.qwen3_safetensor.shaders.embedding_lookup_bf16_f32_sequence import (
    EMBEDDING_LOOKUP_BF16_F32,
)
from torch2vk.models.qwen3_safetensor.shaders.swiglu_f32 import SWIGLU_F32
from torch2vk.vulkan_backend import VulkanContext, create_compute_context
from torch2vk.vulkan_runner import VulkanShaderDispatch


SHADER_DIR = Path("build/shaders/qwen3_safetensor")


def main() -> int:
    context = create_compute_context()
    try:
        _run_embedding(context)
        _run_swiglu(context)
    finally:
        context.close()
    print("qwen3_runner_dispatch=ok shaders=2")
    return 0


def _run_embedding(context: VulkanContext) -> None:
    variant = EMBEDDING_LOOKUP_BF16_F32
    batch = 1
    steps = 2
    hidden = 4
    vocab = 4
    input_ids = (2, 1)
    weights = tuple(float(index) / 4.0 for index in range(vocab * hidden))
    expected = _embedding_reference(weights=weights, input_ids=input_ids, steps=steps, hidden=hidden)
    tensors = {
        "input_ids": input_tensor("runner.input_ids", dtype="int32", shape=(batch, steps)),
        "weight": weight_tensor(
            "runner.embed_tokens",
            dtype="bfloat16",
            shape=(vocab, hidden),
            source_key="smoke.weight",
        ),
        "output": output_tensor("runner.embedding", dtype="float32", shape=(batch, steps, hidden)),
    }
    output = context.create_host_buffer(nbytes=len(expected) * 4)
    ids = context.create_host_buffer(nbytes=len(input_ids) * 4)
    weight = context.create_host_buffer(nbytes=len(weights) * 2)
    ids.write(struct.pack(f"<{len(input_ids)}i", *input_ids))
    weight.write(_pack_bf16(weights))
    dispatch = VulkanShaderDispatch.load(context, variant, shader_dir=SHADER_DIR)
    try:
        dispatch.run(
            tensors=tensors,
            tensor_buffers={
                "runner.input_ids": ids,
                "runner.embed_tokens": weight,
                "runner.embedding": output,
            },
        )
        actual = struct.unpack(f"<{len(expected)}f", output.read(nbytes=len(expected) * 4))
        _assert_close(actual, expected, tolerance=1e-6)
    finally:
        dispatch.close()
        weight.close()
        ids.close()
        output.close()


def _run_swiglu(context: VulkanContext) -> None:
    variant = SWIGLU_F32
    batch = 1
    steps = 1
    width = 4
    gate = (0.0, 1.0, -1.0, 2.0)
    up = (1.0, 2.0, 3.0, 4.0)
    expected = tuple(_silu(a) * b for a, b in zip(gate, up, strict=True))
    tensors = {
        "gate": activation_tensor("runner.gate", dtype="float32", shape=(batch, steps, width)),
        "up": activation_tensor("runner.up", dtype="float32", shape=(batch, steps, width)),
        "output": activation_tensor("runner.swiglu", dtype="float32", shape=(batch, steps, width)),
    }
    gate_buffer = context.create_host_buffer(nbytes=len(gate) * 4)
    up_buffer = context.create_host_buffer(nbytes=len(up) * 4)
    output = context.create_host_buffer(nbytes=len(expected) * 4)
    gate_buffer.write(struct.pack(f"<{len(gate)}f", *gate))
    up_buffer.write(struct.pack(f"<{len(up)}f", *up))
    dispatch = VulkanShaderDispatch.load(context, variant, shader_dir=SHADER_DIR)
    try:
        dispatch.run(
            tensors=tensors,
            tensor_buffers={
                "runner.gate": gate_buffer,
                "runner.up": up_buffer,
                "runner.swiglu": output,
            },
        )
        actual = struct.unpack(f"<{len(expected)}f", output.read(nbytes=len(expected) * 4))
        _assert_close(actual, expected, tolerance=1e-5)
    finally:
        dispatch.close()
        output.close()
        up_buffer.close()
        gate_buffer.close()


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


def _silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def _assert_close(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    *,
    tolerance: float,
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"length mismatch actual={len(actual)} expected={len(expected)}")
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
        if abs(actual_value - expected_value) > tolerance:
            raise AssertionError(
                f"value mismatch at {index}: actual={actual_value} expected={expected_value}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
