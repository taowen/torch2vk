"""Reduce token-score partials in independent chunks."""

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


QWEN3_TOKEN_SELECT_REDUCE_CHUNKS_F32 = ShaderVariant(
    name="qwen3_token_select_reduce_chunks_f32",
    family="quantized_qwen3_asr",
    contract=ShaderContract(
        class_name="Qwen3TokenSelectReduceChunksF32Program",
        shader_name="qwen3_token_select_reduce_chunks_f32",
        fields=(
            TensorFieldSpec(
                "scores", IOKind.INPUT, "scores", TensorContract(dtype="float32", shape=("G",))
            ),
            TensorFieldSpec(
                "tokens", IOKind.INPUT, "tokens", TensorContract(dtype="uint32", shape=("G",))
            ),
            TensorFieldSpec(
                "chunk_scores",
                IOKind.OUTPUT,
                "chunk_scores",
                TensorContract(dtype="float32", shape=("C",)),
            ),
            TensorFieldSpec(
                "chunk_tokens",
                IOKind.OUTPUT,
                "chunk_tokens",
                TensorContract(dtype="uint32", shape=("C",)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(PushConstantFieldSpec("G", PushConstantType.UINT32, 0, "G"),),
        ),
        dispatch=(ceil_div("G", 1024), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
    ),
    source="""\
#version 450

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly ScoresBuffer { float scores[]; };
layout(set = 0, binding = 1) buffer restrict readonly TokensBuffer { uint tokens[]; };
layout(set = 0, binding = 2) buffer restrict writeonly ChunkScoresBuffer { float chunk_scores[]; };
layout(set = 0, binding = 3) buffer restrict writeonly ChunkTokensBuffer { uint chunk_tokens[]; };

layout(push_constant) uniform PushConstants { uint G; } pc;

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

shared float shared_score[16];
shared uint shared_token[16];

bool better_pair(float lhs_score, uint lhs_token, float rhs_score, uint rhs_token) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_token < rhs_token);
}

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint index = gl_WorkGroupID.x * 1024u + tid;
    float best_score = -3.4028234663852886e+38;
    uint best_token = 0xffffffffu;
    if (index < pc.G) {
        best_score = scores[index];
        best_token = tokens[index];
    }

    const float subgroup_best_score = subgroupMax(best_score);
    uint subgroup_best_token = best_token;
    if (best_score != subgroup_best_score) {
        subgroup_best_token = 0xffffffffu;
    }
    subgroup_best_token = subgroupMin(subgroup_best_token);

    const uint subgroup_id = tid >> 6u;
    const uint subgroup_lane = gl_SubgroupInvocationID;
    if (subgroup_lane == 0u) {
        shared_score[subgroup_id] = subgroup_best_score;
        shared_token[subgroup_id] = subgroup_best_token;
    }
    barrier();

    if (tid == 0u) {
        float score = shared_score[0];
        uint token = shared_token[0];
        for (uint i = 1u; i < 16u; ++i) {
            if (better_pair(shared_score[i], shared_token[i], score, token)) {
                score = shared_score[i];
                token = shared_token[i];
            }
        }
        chunk_scores[gl_WorkGroupID.x] = score;
        chunk_tokens[gl_WorkGroupID.x] = token;
    }
}
""",
)
