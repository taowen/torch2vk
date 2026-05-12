"""Final-stage greedy token reduction for Qwen3 logits."""

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
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
    SubgroupRequirements,
)


QWEN3_TOKEN_SELECT_REDUCE_F32 = ShaderVariant(
    name="qwen3_token_select_reduce_f32",
    family="optimized_qwen3",
    contract=ShaderContract(
        class_name="Qwen3TokenSelectReduceF32Program",
        shader_name="qwen3_token_select_reduce_f32",
        fields=(
            TensorFieldSpec("partial_scores", IOKind.INPUT, "partial_scores", TensorContract(dtype="float32", shape=("G",))),
            TensorFieldSpec("partial_tokens", IOKind.INPUT, "partial_tokens", TensorContract(dtype="uint32", shape=("G",))),
            TensorFieldSpec("eos_token_ids", IOKind.INPUT, "eos_token_ids", TensorContract(dtype="int64", shape=("E",))),
            TensorFieldSpec("next_token", IOKind.OUTPUT, "next_token", TensorContract(dtype="int64", shape=(1, 1))),
            TensorFieldSpec("done", IOKind.OUTPUT, "done", TensorContract(dtype="uint32", shape=(1,))),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("G", PushConstantType.UINT32, 0, "G"),
                PushConstantFieldSpec("E", PushConstantType.UINT32, 4, "E"),
            ),
        ),
        dispatch=(1, 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        require_shader_int64=True,
    ),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly PartialScoresBuffer { float partial_scores[]; };
layout(set = 0, binding = 1) buffer restrict readonly PartialTokensBuffer { uint partial_tokens[]; };
layout(set = 0, binding = 2) buffer restrict readonly EosBuffer { int64_t eos_token_ids[]; };
layout(set = 0, binding = 3) buffer restrict writeonly NextTokenBuffer { int64_t next_token[]; };
layout(set = 0, binding = 4) buffer restrict writeonly DoneBuffer { uint done[]; };

layout(push_constant) uniform PushConstants { uint G; uint E; } pc;

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

shared float shared_score[16];
shared uint shared_token[16];

bool better_pair(float lhs_score, uint lhs_token, float rhs_score, uint rhs_token) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_token < rhs_token);
}

void main() {
    const uint tid = gl_LocalInvocationID.x;
    float best_score = -3.4028234663852886e+38;
    uint best_token = 0xffffffffu;
    if (tid < pc.G) {
        best_score = partial_scores[tid];
        best_token = partial_tokens[tid];
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
        next_token[0] = int64_t(token);
        uint is_done = 0u;
        for (uint i = 0u; i < pc.E; ++i) {
            if (int64_t(token) == eos_token_ids[i]) {
                is_done = 1u;
            }
        }
        done[0] = is_done;
    }
}
""",
)
