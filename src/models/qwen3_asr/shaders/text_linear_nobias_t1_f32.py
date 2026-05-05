"""Qwen3-ASR BF16-weight linear shader for one-token decode."""

from __future__ import annotations

from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderExecutionRequirements,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
    ceil_div,
)


QWEN3_ASR_TEXT_LINEAR_NOBIAS_T1_F32 = ShaderVariant(
    name="qwen3_asr_text_linear_nobias_t1_f32",
    family="qwen3_asr.text",
    contract=ShaderContract(
        class_name="Qwen3AsrTextLinearNobiasT1F32Program",
        shader_name="qwen3_asr_text_linear_nobias_t1_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=(1, 1, "K")),
            ),
            TensorFieldSpec(
                name="weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(dtype="bfloat16", shape=("N", "K")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=(1, 1, "N")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("K", PushConstantType.UINT32, 0, "K"),
                PushConstantFieldSpec("N", PushConstantType.UINT32, 4, "N"),
            ),
        ),
        dispatch=(ceil_div("N", 16), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer {
    float x[];
};

layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer {
    uint16_t weight[];
};

layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer {
    float output_values[];
};

layout(push_constant) uniform PushConstants {
    uint K;
    uint N;
} pc;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

shared float partial[16 * 16];

float bf16_to_f32(uint16_t value) {
    return uintBitsToFloat(uint(value) << 16);
}

void main() {
    const uint k_lane = gl_LocalInvocationID.x;
    const uint local_col = gl_LocalInvocationID.y;
    const uint col = gl_WorkGroupID.x * 16u + local_col;
    const uint lane = local_col * 16u + k_lane;

    float acc = 0.0;
    if (col < pc.N) {
        for (uint k = k_lane; k < pc.K; k += 16u) {
            acc += x[k] * bf16_to_f32(weight[col * pc.K + k]);
        }
    }
    partial[lane] = acc;
    barrier();

    if (k_lane < 8u) {
        partial[lane] += partial[lane + 8u];
    }
    barrier();
    if (k_lane < 4u) {
        partial[lane] += partial[lane + 4u];
    }
    barrier();
    if (k_lane < 2u) {
        partial[lane] += partial[lane + 2u];
    }
    barrier();
    if (k_lane == 0u) {
        partial[lane] += partial[lane + 1u];
        if (col < pc.N) {
            output_values[col] = partial[lane];
        }
    }
}
""".lstrip(),
)

