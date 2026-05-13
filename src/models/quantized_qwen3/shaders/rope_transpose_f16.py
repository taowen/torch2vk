"""Fused rotate-half RoPE and transpose for Qwen3 attention tensors."""

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


ROPE_TRANSPOSE_F16 = ShaderVariant(
    name="rope_transpose_f16",
    family="quantized_qwen3",
    contract=ShaderContract(
        class_name="QuantizedQwen3RopeTransposeF16Program",
        shader_name="rope_transpose_f16",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float16", shape=("B", "T", "G", "D")),
            ),
            TensorFieldSpec(
                name="cos",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float16", shape=("B", "T", "D")),
            ),
            TensorFieldSpec(
                name="sin",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float16", shape=("B", "T", "D")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float16", shape=("B", "G", "T", "D")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec("N", PushConstantType.UINT32, 0, mul(mul(mul("B", "T"), "G"), "D")),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
                PushConstantFieldSpec("G", PushConstantType.UINT32, 8, "G"),
                PushConstantFieldSpec("D", PushConstantType.UINT32, 12, "D"),
            ),
        ),
        dispatch=(ceil_div(mul(mul(mul("B", "T"), "G"), "D"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly CosBuffer { float16_t cos_values[]; };
layout(set = 0, binding = 2) buffer restrict readonly SinBuffer { float16_t sin_values[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };

layout(push_constant) uniform PushConstants { uint N; uint T; uint G; uint D; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) {
        return;
    }

    const uint d = idx % pc.D;
    const uint t = (idx / pc.D) % pc.T;
    const uint g = (idx / (pc.D * pc.T)) % pc.G;
    const uint b = idx / (pc.D * pc.T * pc.G);
    const uint half_d = pc.D >> 1u;
    const uint rotated_d = d < half_d ? d + half_d : d - half_d;

    const uint x_base = ((b * pc.T + t) * pc.G + g) * pc.D;
    const uint rope_idx = (b * pc.T + t) * pc.D + d;
    const float16_t x_value = x[x_base + d];
    const float16_t rotated = d < half_d ? -x[x_base + rotated_d] : x[x_base + rotated_d];
    const float16_t left = float16_t(x_value * cos_values[rope_idx]);
    const float16_t right = float16_t(rotated * sin_values[rope_idx]);
    output_values[((b * pc.G + g) * pc.T + t) * pc.D + d] = float16_t(left + right);
}
""",
)
