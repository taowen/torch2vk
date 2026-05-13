"""Q6_K lm_head matvec that emits local argmax partials."""

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
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
    SubgroupRequirements,
)
from torch2vk.vulkan.types import q6_k_halfwords_layout


LM_HEAD_Q6_K_ARGMAX_PARTIAL_F32 = ShaderVariant(
    name="lm_head_q6_k_argmax_partial_f32",
    family="qwen3.text",
    contract=ShaderContract(
        class_name="LmHeadQ6KArgmaxPartialF32Program",
        shader_name="lm_head_q6_k_argmax_partial_f32",
        fields=(
            TensorFieldSpec(
                "x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=(1, 1, "K"))
            ),
            TensorFieldSpec(
                "weight",
                IOKind.INPUT,
                "weight",
                TensorContract(
                    dtype="uint16",
                    shape=("N", mul(ceil_div("K", 256), 105)),
                    layout=q6_k_halfwords_layout(logical_k="K"),
                ),
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

#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 0) buffer restrict readonly XVec4Buffer { vec4 x4[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly PartialScoresBuffer { float partial_scores[]; };
layout(set = 0, binding = 3) buffer restrict writeonly PartialTokensBuffer { uint partial_tokens[]; };

layout(push_constant) uniform PushConstants { uint K; uint N; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared float shared_score[4];
shared uint shared_token[4];

uint q6k_byte(uint block_half, uint byte_offset) {
    const uint packed = uint(weight[block_half + (byte_offset >> 1u)]);
    return ((byte_offset & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
}

uint q6k_u32(uint block_half, uint byte_offset) {
    const uint half_index = block_half + (byte_offset >> 1u);
    return uint(weight[half_index]) | (uint(weight[half_index + 1u]) << 16u);
}

int q6k_i8(uint block_half, uint byte_offset) {
    int value = int(q6k_byte(block_half, byte_offset));
    if (value >= 128) {
        value -= 256;
    }
    return value;
}

vec4 unpack8_f32(uint value) {
    return vec4(
        float(value & 255u),
        float((value >> 8u) & 255u),
        float((value >> 16u) & 255u),
        float((value >> 24u) & 255u)
    );
}

vec4 load_x4(uint k) {
    return x4[k >> 2u];
}

float q6k_block_dot(uint col, uint block_index, uint itid) {
    const uint blocks_per_row = pc.K / 256u;
    const uint block_half = col * blocks_per_row * 105u + block_index * 105u;
    const uint k_base = block_index * 256u;

    const uint v_im = itid >> 3u;
    const uint v_in = itid - 8u * v_im;
    const uint l0 = 4u * v_in;
    const uint is = v_in >> 2u;

    const uint ql_offset = 64u * v_im + l0;
    const uint qh_offset = 128u + 32u * v_im + l0;
    const uint scale_offset = 192u + 8u * v_im + is;
    const uint x_offset = k_base + 128u * v_im + l0;

    const uint ql0 = q6k_u32(block_half, ql_offset);
    const uint ql32 = q6k_u32(block_half, ql_offset + 32u);
    const uint qh = q6k_u32(block_half, qh_offset);

    const uint q0_word = (ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4u);
    const uint q1_word = (ql32 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2u);
    const uint q2_word = ((ql0 >> 4u) & 0x0F0F0F0Fu) | (qh & 0x30303030u);
    const uint q3_word = ((ql32 >> 4u) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2u);

    const vec4 q0 = unpack8_f32(q0_word) - vec4(32.0);
    const vec4 q1 = unpack8_f32(q1_word) - vec4(32.0);
    const vec4 q2 = unpack8_f32(q2_word) - vec4(32.0);
    const vec4 q3 = unpack8_f32(q3_word) - vec4(32.0);

    const vec4 x0 = load_x4(x_offset);
    const vec4 x1 = load_x4(x_offset + 32u);
    const vec4 x2 = load_x4(x_offset + 64u);
    const vec4 x3 = load_x4(x_offset + 96u);

    const float d = unpackHalf2x16(uint(weight[block_half + 104u])).x;
    const float s0 = float(q6k_i8(block_half, scale_offset));
    const float s1 = float(q6k_i8(block_half, scale_offset + 2u));
    const float s2 = float(q6k_i8(block_half, scale_offset + 4u));
    const float s3 = float(q6k_i8(block_half, scale_offset + 6u));
    return d * (dot(q0, x0) * s0 + dot(q1, x1) * s1 + dot(q2, x2) * s2 + dot(q3, x3) * s3);
}

float q6k_dot(uint col) {
    const uint lane = gl_SubgroupInvocationID;
    const uint itid = lane & 15u;
    const uint ix = lane >> 4u;
    const uint blocks_per_row = pc.K / 256u;
    float acc = 0.0;
    for (uint block = ix; block < blocks_per_row; block += 4u) {
        acc += q6k_block_dot(col, block, itid);
    }
    return subgroupAdd(acc);
}

bool better_pair(float lhs_score, uint lhs_token, float rhs_score, uint rhs_token) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_token < rhs_token);
}

void main() {
    const uint subgroup_id = gl_SubgroupID;
    const uint subgroup_lane = gl_SubgroupInvocationID;
    const uint token = gl_WorkGroupID.x * 4u + subgroup_id;
    const float score = token < pc.N ? q6k_dot(token) : -3.4028234663852886e+38;
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
