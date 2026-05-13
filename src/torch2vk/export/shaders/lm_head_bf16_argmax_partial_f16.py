"""BF16 lm_head matvec that emits local argmax partials."""

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
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
    SubgroupRequirements,
)


LM_HEAD_BF16_ARGMAX_PARTIAL_F16 = ShaderVariant(
    name="lm_head_bf16_argmax_partial_f16",
    family="qwen3.text",
    contract=ShaderContract(
        class_name="LmHeadBf16ArgmaxPartialF16Program",
        shader_name="lm_head_bf16_argmax_partial_f16",
        fields=(
            TensorFieldSpec(
                "x", IOKind.INPUT, "input", TensorContract(dtype="float16", shape=(1, 1, "K"))
            ),
            TensorFieldSpec(
                "weight",
                IOKind.INPUT,
                "weight",
                TensorContract(dtype="bfloat16", shape=("N", "K")),
            ),
            TensorFieldSpec(
                "partial_scores",
                IOKind.OUTPUT,
                "partial_scores",
                TensorContract(dtype="float32", shape=("G",)),
            ),
            TensorFieldSpec(
                "partial_tokens",
                IOKind.OUTPUT,
                "partial_tokens",
                TensorContract(dtype="uint32", shape=("G",)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("K", PushConstantType.UINT32, 0, "K"),
                PushConstantFieldSpec("N", PushConstantType.UINT32, 4, "N"),
            ),
        ),
        dispatch=(ceil_div("N", 4), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        require_storage_buffer_16bit_access=True,
    ),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_bfloat16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { bfloat16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly PartialScoresBuffer { float partial_scores[]; };
layout(set = 0, binding = 3) buffer restrict writeonly PartialTokensBuffer { uint partial_tokens[]; };

layout(push_constant) uniform PushConstants { uint K; uint N; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared float shared_score[4];
shared uint shared_token[4];

bool better_pair(float lhs_score, uint lhs_token, float rhs_score, uint rhs_token) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_token < rhs_token);
}

float token_score(uint token) {
    const uint lane = gl_SubgroupInvocationID;
    float acc = 0.0;
    for (uint k = lane; k < pc.K; k += 64u) {
        acc = fma(float(x[k]), float(weight[token * pc.K + k]), acc);
    }
    return subgroupAdd(acc);
}

void main() {
    const uint subgroup_id = gl_SubgroupID;
    const uint subgroup_lane = gl_SubgroupInvocationID;
    const uint token = gl_WorkGroupID.x * 4u + subgroup_id;
    const float score = token < pc.N ? token_score(token) : -3.4028234663852886e+38;
    if (subgroup_lane == 0u) {
        shared_score[subgroup_id] = score;
        shared_token[subgroup_id] = token;
    }
    barrier();

    if (gl_LocalInvocationID.x == 0u) {
        float best_score = shared_score[0];
        uint best_token = shared_token[0];
        for (uint i = 1u; i < 4u; ++i) {
            if (better_pair(shared_score[i], shared_token[i], best_score, best_token)) {
                best_score = shared_score[i];
                best_token = shared_token[i];
            }
        }
        partial_scores[gl_WorkGroupID.x] = best_score;
        partial_tokens[gl_WorkGroupID.x] = best_token;
    }
}
""",
)
