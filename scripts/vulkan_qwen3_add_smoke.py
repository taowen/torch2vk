#!/usr/bin/env python3
"""Run the copied Qwen3 F32 add shader through Vulkan."""

from __future__ import annotations

import struct
from pathlib import Path

from torch2vk.logical import activation_tensor
from torch2vk.models.qwen3_safetensor.shaders.add_f32 import ADD_F32
from torch2vk.shader import pack_push_constants
from torch2vk.vulkan_backend import create_compute_context


def main() -> int:
    variant = ADD_F32
    shape = (1, 2, 4)
    lhs = tuple(float(index) for index in range(8))
    rhs = tuple(float(index) * 10.0 for index in range(8))
    expected = tuple(a + b for a, b in zip(lhs, rhs, strict=True))
    tensors = {
        "lhs": activation_tensor("lhs", dtype="float32", shape=shape),
        "rhs": activation_tensor("rhs", dtype="float32", shape=shape),
        "output": activation_tensor("output", dtype="float32", shape=shape),
    }
    symbols = variant.contract.validate(tensors)

    context = create_compute_context()
    try:
        lhs_buffer = context.create_host_buffer(nbytes=len(lhs) * 4)
        rhs_buffer = context.create_host_buffer(nbytes=len(rhs) * 4)
        output_buffer = context.create_host_buffer(nbytes=len(expected) * 4)
        partial_buffer = context.create_host_buffer(nbytes=4)
        lhs_buffer.write(struct.pack(f"<{len(lhs)}f", *lhs))
        rhs_buffer.write(struct.pack(f"<{len(rhs)}f", *rhs))

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
                    0: lhs_buffer,
                    1: rhs_buffer,
                    2: output_buffer,
                    3: partial_buffer,
                },
                descriptor_types={
                    0: "storage_buffer",
                    1: "storage_buffer",
                    2: "storage_buffer",
                    3: "storage_buffer",
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
            command_buffer.dispatch(shape[2], shape[1], shape[0])
            command_buffer.end()
            command_buffer.submit_and_wait(fence)

            actual = struct.unpack(
                f"<{len(expected)}f",
                output_buffer.read(nbytes=len(expected) * 4),
            )
            if actual != expected:
                raise AssertionError(f"add mismatch actual={actual} expected={expected}")
            print(f"qwen3_add_dispatch=ok values={actual}")
        finally:
            fence.close()
            command_pool.close()
            pipeline.close()
            pipeline_layout.close()
            descriptor_pool.close()
            descriptor_layout.close()
            module.close()
            partial_buffer.close()
            output_buffer.close()
            rhs_buffer.close()
            lhs_buffer.close()
    finally:
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
