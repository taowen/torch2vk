#!/usr/bin/env python3
"""Run the Qwen3 flash-attention shader through Vulkan."""

from __future__ import annotations

import struct
from pathlib import Path

from torch2vk.logical import activation_tensor
from torch2vk.models.qwen3_safetensor.shaders.fa_split_k_reduce import FA_SPLIT_K_REDUCE
from torch2vk.models.qwen3_safetensor.shaders.flash_attn_f32_f16 import FLASH_ATTN_F32_F16
from torch2vk.shader import pack_push_constants
from torch2vk.vulkan_backend import VulkanBuffer, create_compute_context


def main() -> int:
    batch = 1
    steps = 1
    cache_steps = 64
    q_heads = 1
    kv_heads = 1
    head_dim = 128
    split_k_floats = (head_dim * q_heads + 2 * q_heads) * 4
    q = tuple(0.0 for _ in range(head_dim))
    k = tuple(0.0 for _ in range(cache_steps * head_dim))
    v_row = tuple(float((index % 7) + 1) / 8.0 for index in range(head_dim))
    v = tuple(value for _row in range(cache_steps) for value in v_row)
    expected = tuple(_f16_round(value) for value in v_row)

    flash = FLASH_ATTN_F32_F16
    reduce = FA_SPLIT_K_REDUCE
    flash_tensors = {
        "q": activation_tensor("q", dtype="float32", shape=(batch, steps, q_heads, head_dim)),
        "k": activation_tensor(
            "k",
            dtype="float16",
            shape=(batch, cache_steps, kv_heads, head_dim),
        ),
        "v": activation_tensor(
            "v",
            dtype="float16",
            shape=(batch, cache_steps, kv_heads, head_dim),
        ),
        "mask": activation_tensor(
            "mask",
            dtype="float16",
            shape=(batch, 1, steps, cache_steps),
        ),
        "sinks_placeholder": activation_tensor(
            "sinks_placeholder",
            dtype="float32",
            shape=(batch, steps, q_heads, head_dim),
        ),
        "mask_opt_placeholder": activation_tensor(
            "mask_opt_placeholder",
            dtype="float32",
            shape=(batch, steps, q_heads, head_dim),
        ),
        "split_k_output": activation_tensor(
            "split_k_output",
            dtype="float32",
            shape=(split_k_floats,),
        ),
    }
    reduce_tensors = {
        "split_k_input": flash_tensors["split_k_output"],
        "sinks_placeholder": flash_tensors["sinks_placeholder"],
        "output": activation_tensor(
            "output",
            dtype="float32",
            shape=(batch, steps, q_heads, head_dim),
        ),
    }
    flash_symbols = flash.contract.validate(flash_tensors)
    reduce_symbols = reduce.contract.validate(reduce_tensors)

    context = create_compute_context()
    try:
        q_buffer = context.create_host_buffer(nbytes=len(q) * 4)
        k_buffer = context.create_host_buffer(nbytes=len(k) * 2)
        v_buffer = context.create_host_buffer(nbytes=len(v) * 2)
        mask_buffer = context.create_host_buffer(nbytes=cache_steps * 2)
        placeholder_buffer = context.create_host_buffer(nbytes=head_dim * 4)
        split_buffer = context.create_host_buffer(nbytes=split_k_floats * 4)
        output_buffer = context.create_host_buffer(nbytes=len(expected) * 4)
        q_buffer.write(struct.pack(f"<{len(q)}f", *q))
        k_buffer.write(struct.pack(f"<{len(k)}e", *k))
        v_buffer.write(struct.pack(f"<{len(v)}e", *v))
        mask_buffer.write(struct.pack(f"<{cache_steps}e", *(0.0 for _ in range(cache_steps))))
        placeholder_buffer.write(bytes(head_dim * 4))
        split_buffer.write(bytes(split_k_floats * 4))
        output_buffer.write(bytes(len(expected) * 4))

        _dispatch_flash(
            context=context,
            flash_tensors=flash_tensors,
            flash_symbols=flash_symbols,
            buffers={
                0: q_buffer,
                1: k_buffer,
                2: v_buffer,
                3: mask_buffer,
                4: placeholder_buffer,
                5: split_buffer,
                6: placeholder_buffer,
            },
        )
        _dispatch_reduce(
            context=context,
            reduce_tensors=reduce_tensors,
            reduce_symbols=reduce_symbols,
            buffers={
                0: split_buffer,
                1: placeholder_buffer,
                2: output_buffer,
            },
        )

        actual = struct.unpack(
            f"<{len(expected)}f",
            output_buffer.read(nbytes=len(expected) * 4),
        )
        _assert_close(actual, expected)
        print(f"qwen3_flash_attn_dispatch=ok values={actual[:4]}")
        output_buffer.close()
        split_buffer.close()
        placeholder_buffer.close()
        mask_buffer.close()
        v_buffer.close()
        k_buffer.close()
        q_buffer.close()
    finally:
        context.close()
    return 0


def _dispatch_flash(
    *,
    context,
    flash_tensors,
    flash_symbols,
    buffers: dict[int, VulkanBuffer],
) -> None:
    variant = FLASH_ATTN_F32_F16
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
            buffers,
            descriptor_types={binding: "storage_buffer" for binding in buffers},
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
            data=pack_push_constants(variant.contract, flash_tensors, flash_symbols) or b"",
        )
        command_buffer.dispatch(4, 1, 1)
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


def _dispatch_reduce(
    *,
    context,
    reduce_tensors,
    reduce_symbols,
    buffers: dict[int, VulkanBuffer],
) -> None:
    variant = FA_SPLIT_K_REDUCE
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
            buffers,
            descriptor_types={binding: "storage_buffer" for binding in buffers},
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
            data=pack_push_constants(variant.contract, reduce_tensors, reduce_symbols) or b"",
        )
        command_buffer.dispatch(1, 2, 1)
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
