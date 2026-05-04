"""Qwen3-ASR cached single-token decode attention with GQA."""

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
)
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements, SubgroupRequirements


QWEN3_ASR_TEXT_ATTENTION_DECODE_F32 = ShaderVariant(
    name="qwen3_asr_text_attention_decode_f32",
    family="qwen3_asr.text",
    contract=ShaderContract(
        class_name="Qwen3AsrTextAttentionDecodeF32Program",
        shader_name="qwen3_asr_text_attention_decode_f32",
        fields=(
            TensorFieldSpec(
                name="q",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=(1, 1, "QH")),
            ),
            TensorFieldSpec(
                name="key_cache",
                io_kind=IOKind.INPUT,
                role="state",
                contract=TensorContract(dtype="float32", shape=(1, "NK", "S", "D")),
            ),
            TensorFieldSpec(
                name="value_cache",
                io_kind=IOKind.INPUT,
                role="state",
                contract=TensorContract(dtype="float32", shape=(1, "NK", "S", "D")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=(1, 1, "QH")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=24,
            fields=(
                PushConstantFieldSpec("num_q_heads", PushConstantType.UINT32, 0, ceil_div("QH", "D")),
                PushConstantFieldSpec("num_kv_heads", PushConstantType.UINT32, 4, "NK"),
                PushConstantFieldSpec("head_dim", PushConstantType.UINT32, 8, "D"),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 12, "S"),
                PushConstantFieldSpec("cache_len", PushConstantType.UINT32, 16, "S"),
                PushConstantFieldSpec("QH", PushConstantType.UINT32, 20, "QH"),
            ),
        ),
        dispatch=(ceil_div("QH", "D"), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True)
    ),
    source="""
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly QBuffer {
    float q_values[];
};

layout(set = 0, binding = 1) buffer restrict readonly KeyCacheBuffer {
    float key_cache[];
};

layout(set = 0, binding = 2) buffer restrict readonly ValueCacheBuffer {
    float value_cache[];
};

layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer {
    float output_values[];
};

layout(push_constant) uniform PushConstants {
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint S;
    uint cache_len;
    uint QH;
} pc;

// One workgroup per Q head. Each thread handles one head dimension.
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

const float NEG_INF = -3.4028234663852886e38;

shared float subgroup_dot[4];

void main() {
    const uint q_head = gl_WorkGroupID.x;
    const uint dim = gl_LocalInvocationID.x;
    if (q_head >= pc.num_q_heads) {
        return;
    }
    const bool valid_dim = dim < pc.head_dim;

    const uint kv_head = q_head * pc.num_kv_heads / pc.num_q_heads;
    const float q_value = valid_dim ? q_values[q_head * pc.head_dim + dim] : 0.0;
    const float scale = inversesqrt(float(pc.head_dim));

    float running_max = NEG_INF;
    float running_sum = 0.0;
    float acc = 0.0;

    const uint cache_head_base = kv_head * pc.S * pc.head_dim;
    for (uint key_pos = 0u; key_pos < pc.cache_len; ++key_pos) {
        const float k_val = valid_dim ? key_cache[cache_head_base + key_pos * pc.head_dim + dim] : 0.0;
        const float v_val = valid_dim ? value_cache[cache_head_base + key_pos * pc.head_dim + dim] : 0.0;
        barrier();

        const float dot_part = valid_dim ? q_value * k_val : 0.0;
        const float dot_sum = subgroupAdd(dot_part);
        if (gl_SubgroupInvocationID == 0u) {
            subgroup_dot[dim / gl_SubgroupSize] = dot_sum;
        }
        barrier();

        if (valid_dim) {
            float dot = 0.0;
            for (uint i = 0u; i < (pc.head_dim + gl_SubgroupSize - 1u) / gl_SubgroupSize; ++i) {
                dot += subgroup_dot[i];
            }
            const float score = dot * scale;
            const float next_max = max(running_max, score);
            const float old_scale_f = running_max == NEG_INF ? 0.0 : exp(running_max - next_max);
            const float score_scale = exp(score - next_max);
            acc = acc * old_scale_f + score_scale * v_val;
            running_sum = running_sum * old_scale_f + score_scale;
            running_max = next_max;
        }
        barrier();
    }

    if (valid_dim && running_sum > 0.0) {
        output_values[q_head * pc.head_dim + dim] = acc / running_sum;
    }
}
""".lstrip(),
)
