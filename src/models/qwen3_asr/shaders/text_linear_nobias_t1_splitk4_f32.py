"""Qwen3-ASR BF16-weight single-token linear shader with four-way K split."""

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
)
from torch2vk.vulkan.shader_execution_requirements import SubgroupRequirements


QWEN3_ASR_TEXT_LINEAR_NOBIAS_T1_SPLITK4_F32 = ShaderVariant(
    name="qwen3_asr_text_linear_nobias_t1_splitk4_f32",
    family="qwen3_asr.text",
    contract=ShaderContract(
        class_name="Qwen3AsrTextLinearNobiasT1Splitk4F32Program",
        shader_name="qwen3_asr_text_linear_nobias_t1_splitk4_f32",
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
        dispatch=("N", 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        require_storage_buffer_16bit_access=True,
    ),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

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

layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;

shared float part_score[4];

float bf16_to_f32(uint16_t value) {
    return uintBitsToFloat(uint(value) << 16);
}

void main() {
    const uint k_lane = gl_LocalInvocationID.x;
    const uint part = gl_LocalInvocationID.y;
    const uint col = gl_WorkGroupID.x;

    float acc = 0.0;
    if (col < pc.N) {
        for (uint k = k_lane + part * 64u; k < pc.K; k += 256u) {
            acc += x[k] * bf16_to_f32(weight[col * pc.K + k]);
        }
    }

    const float score = subgroupAdd(acc);
    if (gl_SubgroupInvocationID == 0u) {
        part_score[part] = score;
    }
    barrier();

    if (k_lane == 0u && part == 0u && col < pc.N) {
        output_values[col] = part_score[0] + part_score[1] + part_score[2] + part_score[3];
    }
}
""".lstrip(),
)
