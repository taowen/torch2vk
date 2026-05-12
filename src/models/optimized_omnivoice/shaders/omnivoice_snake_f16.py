"""Fused Snake activation shader for optimized OmniVoice audio decode."""

from __future__ import annotations

from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
    ceil_div,
    mul,
)
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements


OMNIVOICE_SNAKE_F16 = ShaderVariant(
    name="omnivoice_snake_f16",
    family="optimized_omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceSnakeF16Program",
        shader_name="omnivoice_snake_f16",
        fields=(
            TensorFieldSpec(
                name="alpha",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=(1, "C", 1)),
            ),
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float16", shape=(1, "C", "T")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float16", shape=(1, "C", "T")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("C", PushConstantType.UINT32, 0, "C", dynamic=False),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T", dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul("C", "T"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly AlphaBuffer { float alpha[]; };
layout(set = 0, binding = 1) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };

layout(push_constant) uniform PushConstants { uint C; uint T; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.C * pc.T;
    if (idx >= total) {
        return;
    }
    const uint c = idx / pc.T;
    const float a = alpha[c];
    const float value = float(x[idx]);
    const float s = sin(a * value);
    output_values[idx] = float16_t(value + (s * s) / (a + 1e-9));
}
""",
)
