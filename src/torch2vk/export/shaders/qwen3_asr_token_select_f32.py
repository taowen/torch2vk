"""Qwen3-ASR greedy token selection shader."""

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


QWEN3_ASR_TOKEN_SELECT_GREEDY_F32 = ShaderVariant(
    name="qwen3_asr_token_select_greedy_f32",
    family="qwen3_asr.text",
    contract=ShaderContract(
        class_name="Qwen3AsrTokenSelectGreedyF32Program",
        shader_name="qwen3_asr_token_select_greedy_f32",
        fields=(
            TensorFieldSpec(
                name="logits",
                io_kind=IOKind.INPUT,
                role="logits",
                contract=TensorContract(dtype="float16", shape=(1, "T", "V")),
            ),
            TensorFieldSpec(
                name="eos_token_ids",
                io_kind=IOKind.INPUT,
                role="eos_token_ids",
                contract=TensorContract(dtype="int64", shape=("E",)),
            ),
            TensorFieldSpec(
                name="next_token",
                io_kind=IOKind.OUTPUT,
                role="next_token",
                contract=TensorContract(dtype="int64", shape=(1, 1)),
            ),
            TensorFieldSpec(
                name="done",
                io_kind=IOKind.OUTPUT,
                role="done",
                contract=TensorContract(dtype="uint32", shape=(1,)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("T", PushConstantType.UINT32, 0, "T"),
                PushConstantFieldSpec("V", PushConstantType.UINT32, 4, "V"),
                PushConstantFieldSpec("E", PushConstantType.UINT32, 8, "E"),
            ),
        ),
        dispatch=(1, 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        require_shader_int64=True,
        require_storage_buffer_16bit_access=True,
    ),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly LogitsBuffer {
    float16_t logits[];
};

layout(set = 0, binding = 1) buffer restrict readonly EosBuffer {
    int64_t eos_token_ids[];
};

layout(set = 0, binding = 2) buffer restrict writeonly NextTokenBuffer {
    int64_t next_token[];
};

layout(set = 0, binding = 3) buffer restrict writeonly DoneBuffer {
    uint done[];
};

layout(push_constant) uniform PushConstants {
    uint T;
    uint V;
    uint E;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared float shared_score[256];
shared uint shared_token[256];

bool better_pair(float lhs_score, uint lhs_token, float rhs_score, uint rhs_token) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_token < rhs_token);
}

void main() {
    const uint tid = gl_LocalInvocationID.x;
    float best_score = -3.4028234663852886e+38;
    uint best_token = 0u;
    const uint row_offset = (pc.T - 1u) * pc.V;

    for (uint token = tid; token < pc.V; token += 256u) {
        const float score = logits[row_offset + token];
        if (better_pair(score, token, best_score, best_token)) {
            best_score = score;
            best_token = token;
        }
    }

    const float subgroup_best_score = subgroupMax(best_score);
    uint subgroup_best_token = best_token;
    if (best_score != subgroup_best_score) {
        subgroup_best_token = 0xffffffffu;
    }
    subgroup_best_token = subgroupMin(subgroup_best_token);

    const uint subgroup_count = gl_WorkGroupSize.x / gl_SubgroupSize;
    const uint subgroup_id = tid / gl_SubgroupSize;
    const uint subgroup_lane = gl_SubgroupInvocationID;
    if (subgroup_lane == 0u) {
        shared_score[subgroup_id] = subgroup_best_score;
        shared_token[subgroup_id] = subgroup_best_token;
    }
    barrier();

    if (tid < subgroup_count) {
        best_score = shared_score[tid];
        best_token = shared_token[tid];
    } else {
        best_score = -3.4028234663852886e+38;
        best_token = 0xffffffffu;
    }
    barrier();

    for (uint stride = subgroup_count >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float other_score = shared_score[tid + stride];
            const uint other_token = shared_token[tid + stride];
            if (better_pair(other_score, other_token, shared_score[tid], shared_token[tid])) {
                shared_score[tid] = other_score;
                shared_token[tid] = other_token;
            }
        }
        barrier();
    }

    if (tid == 0u) {
        const uint token = shared_token[0];
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
""".lstrip(),
)
