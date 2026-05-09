"""Qwen3-ASR fused decode lm-head greedy selection shaders."""

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
from torch2vk.vulkan.shader_execution_requirements import SubgroupRequirements


QWEN3_ASR_TEXT_LM_HEAD_SELECT_PARTIAL_T1_F32 = ShaderVariant(
    name="qwen3_asr_text_lm_head_select_partial_t1_f32",
    family="qwen3_asr.text",
    contract=ShaderContract(
        class_name="Qwen3AsrTextLmHeadSelectPartialT1F32Program",
        shader_name="qwen3_asr_text_lm_head_select_partial_t1_f32",
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
                name="scratch",
                io_kind=IOKind.OUTPUT,
                role="scratch",
                contract=TensorContract(dtype="float32", shape=("P",)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("K", PushConstantType.UINT32, 0, "K"),
                PushConstantFieldSpec("N", PushConstantType.UINT32, 4, "N"),
                PushConstantFieldSpec("partial_count", PushConstantType.UINT32, 8, ceil_div("N", 4)),
            ),
        ),
        dispatch=(ceil_div("N", 4), 1, 1),
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

layout(set = 0, binding = 2) buffer restrict writeonly ScratchBuffer {
    float scratch[];
};

layout(push_constant) uniform PushConstants {
    uint K;
    uint N;
    uint partial_count;
} pc;

layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;

shared float col_score[4];
shared uint col_token[4];

float bf16_to_f32(uint16_t value) {
    return uintBitsToFloat(uint(value) << 16);
}

bool better_pair(float lhs_score, uint lhs_token, float rhs_score, uint rhs_token) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_token < rhs_token);
}

void main() {
    const uint k_lane = gl_LocalInvocationID.x;
    const uint local_col = gl_LocalInvocationID.y;
    const uint col = gl_WorkGroupID.x * 4u + local_col;

    float acc = 0.0;
    if (col < pc.N) {
        for (uint k = k_lane; k < pc.K; k += 64u) {
            acc += x[k] * bf16_to_f32(weight[col * pc.K + k]);
        }
    }

    const float score = subgroupAdd(acc);
    if (gl_SubgroupInvocationID == 0u) {
        col_score[local_col] = col < pc.N ? score : -3.4028234663852886e+38;
        col_token[local_col] = col < pc.N ? col : 0xffffffffu;
    }
    barrier();

    if (k_lane == 0u && local_col == 0u) {
        float best_score = col_score[0];
        uint best_token = col_token[0];
        for (uint i = 1u; i < 4u; ++i) {
            if (better_pair(col_score[i], col_token[i], best_score, best_token)) {
                best_score = col_score[i];
                best_token = col_token[i];
            }
        }
        scratch[gl_WorkGroupID.x] = best_score;
        scratch[pc.partial_count + gl_WorkGroupID.x] = float(best_token);
    }
}
""".lstrip(),
)


QWEN3_ASR_TEXT_LM_HEAD_SELECT_REDUCE_T1_F32 = ShaderVariant(
    name="qwen3_asr_text_lm_head_select_reduce_t1_f32",
    family=QWEN3_ASR_TEXT_LM_HEAD_SELECT_PARTIAL_T1_F32.family,
    contract=ShaderContract(
        class_name="Qwen3AsrTextLmHeadSelectReduceT1F32Program",
        shader_name="qwen3_asr_text_lm_head_select_reduce_t1_f32",
        fields=(
            TensorFieldSpec(
                name="scratch",
                io_kind=IOKind.INPUT,
                role="scratch",
                contract=TensorContract(dtype="float32", shape=("P",)),
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
            size=8,
            fields=(
                PushConstantFieldSpec("P", PushConstantType.UINT32, 0, "P"),
                PushConstantFieldSpec("E", PushConstantType.UINT32, 4, "E"),
            ),
        ),
        dispatch=(1, 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        require_shader_int64=True,
    ),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly ScratchBuffer {
    float scratch[];
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
    uint P;
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
    const uint partial_count = pc.P >> 1u;
    float best_score = -3.4028234663852886e+38;
    uint best_token = 0xffffffffu;

    for (uint index = tid; index < partial_count; index += 256u) {
        const float score = scratch[index];
        const uint token = uint(scratch[partial_count + index]);
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

    if (tid >= subgroup_count) {
        shared_score[tid] = -3.4028234663852886e+38;
        shared_token[tid] = 0xffffffffu;
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
