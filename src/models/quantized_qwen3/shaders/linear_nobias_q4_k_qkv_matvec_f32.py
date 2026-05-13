"""Fused Q/K/V Q4_K decode matvec for Qwen3 layers whose V is Q4_K."""

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
    add,
    ceil_div,
    mul,
)
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements, SubgroupRequirements
from torch2vk.vulkan.types import q4_k_words_layout


LINEAR_NOBIAS_Q4_K_QKV_MATVEC_F32 = ShaderVariant(
    name="linear_nobias_q4_k_qkv_matvec_f32",
    family="quantized_qwen3",
    contract=ShaderContract(
        class_name="LinearNobiasQ4KQKVMatvecProgram",
        shader_name="linear_nobias_q4_k_qkv_matvec_f32",
        fields=(
            TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float16", shape=("X0", "X1", "K"))),
            TensorFieldSpec(
                "q_weight",
                IOKind.INPUT,
                "weight",
                TensorContract(
                    dtype="uint32",
                    shape=("NQ", mul(ceil_div("K", 256), 36)),
                    layout=q4_k_words_layout(logical_k="K"),
                ),
            ),
            TensorFieldSpec(
                "k_weight",
                IOKind.INPUT,
                "weight",
                TensorContract(
                    dtype="uint32",
                    shape=("NK", mul(ceil_div("K", 256), 36)),
                    layout=q4_k_words_layout(logical_k="K"),
                ),
            ),
            TensorFieldSpec(
                "v_weight",
                IOKind.INPUT,
                "weight",
                TensorContract(
                    dtype="uint32",
                    shape=("NV", mul(ceil_div("K", 256), 36)),
                    layout=q4_k_words_layout(logical_k="K"),
                ),
            ),
            TensorFieldSpec("q_output", IOKind.OUTPUT, "output", TensorContract(dtype="float16", shape=("X0", "X1", "NQ"))),
            TensorFieldSpec("k_output", IOKind.OUTPUT, "output", TensorContract(dtype="float16", shape=("X0", "X1", "NK"))),
            TensorFieldSpec("v_output", IOKind.OUTPUT, "output", TensorContract(dtype="float16", shape=("X0", "X1", "NV"))),
        ),
        push_constants=PushConstantSpec(
            size=20,
            fields=(
                PushConstantFieldSpec("M", PushConstantType.UINT32, 0, mul("X0", "X1")),
                PushConstantFieldSpec("K", PushConstantType.UINT32, 4, "K"),
                PushConstantFieldSpec("NQ", PushConstantType.UINT32, 8, "NQ"),
                PushConstantFieldSpec("NK", PushConstantType.UINT32, 12, "NK"),
                PushConstantFieldSpec("NV", PushConstantType.UINT32, 16, "NV"),
            ),
        ),
        dispatch=(ceil_div(add(add("NQ", "NK"), "NV"), 2), mul("X0", "X1"), 1),
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
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 0) buffer restrict readonly XVec4Buffer { f16vec4 x4[]; };
layout(set = 0, binding = 1) buffer restrict readonly QWeightBuffer { uint q_weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly KWeightBuffer { uint k_weight[]; };
layout(set = 0, binding = 3) buffer restrict readonly VWeightBuffer { uint v_weight[]; };
layout(set = 0, binding = 4) buffer restrict writeonly QOutputBuffer { float16_t q_output[]; };
layout(set = 0, binding = 5) buffer restrict writeonly KOutputBuffer { float16_t k_output[]; };
layout(set = 0, binding = 6) buffer restrict writeonly VOutputBuffer { float16_t v_output[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint NQ; uint NK; uint NV; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

uint total_n() {
    return pc.NQ + pc.NK + pc.NV;
}

uint q4k_byte_q(uint block_word, uint byte_offset) {
    const uint word_value = q_weight[block_word + (byte_offset >> 2u)];
    return (word_value >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

uint q4k_byte_k(uint block_word, uint byte_offset) {
    const uint word_value = k_weight[block_word + (byte_offset >> 2u)];
    return (word_value >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

uint q4k_byte_v(uint block_word, uint byte_offset) {
    const uint word_value = v_weight[block_word + (byte_offset >> 2u)];
    return (word_value >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

uint q4k_u16_q(uint block_word, uint half_index) {
    const uint byte_offset = 4u + half_index * 2u;
    return q4k_byte_q(block_word, byte_offset) | (q4k_byte_q(block_word, byte_offset + 1u) << 8u);
}

uint q4k_u16_k(uint block_word, uint half_index) {
    const uint byte_offset = 4u + half_index * 2u;
    return q4k_byte_k(block_word, byte_offset) | (q4k_byte_k(block_word, byte_offset + 1u) << 8u);
}

uint q4k_u16_v(uint block_word, uint half_index) {
    const uint byte_offset = 4u + half_index * 2u;
    return q4k_byte_v(block_word, byte_offset) | (q4k_byte_v(block_word, byte_offset + 1u) << 8u);
}

vec4 unpack8_f32(uint value) {
    return vec4(
        float(value & 255u),
        float((value >> 8u) & 255u),
        float((value >> 16u) & 255u),
        float((value >> 24u) & 255u)
    );
}

vec4 load_x4(uint row, uint k) {
    return vec4(x4[(row * pc.K + k) >> 2u]);
}

float q4k_block_dot_q(
    uint row,
    uint col,
    uint block_index,
    uint blocks_per_row,
    uint q_offset,
    uint y_offset,
    uint v_im
) {
    const uint block_word = col * blocks_per_row * 36u + block_index * 36u;
    const vec2 dm = unpackHalf2x16(q_weight[block_word]);

    const uint scale0_u32 = q4k_u16_q(block_word, v_im);
    const uint scale4_u32 = q4k_u16_q(block_word, v_im + 2u);
    const uint scale8_u32 = q4k_u16_q(block_word, v_im + 4u);
    const uint scale_0_4_l = (scale4_u32 << 16u) | scale0_u32;
    const uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2u;
    const vec4 scale_0_4_l_f = unpack8_f32(scale_0_4_l & 0x3F3F3F3Fu);
    const vec4 scale8_f = unpack8_f32((((scale8_u32 << 12u) | scale8_u32) & 0x0F0F0F0Fu) | scale_0_4_h);

    const float sc0 = scale_0_4_l_f.x;
    const float sc1 = scale_0_4_l_f.y;
    const float sc2 = scale_0_4_l_f.z;
    const float sc3 = scale_0_4_l_f.w;
    const float sc4 = scale8_f.x;
    const float sc5 = scale8_f.y;
    const float sc6 = scale8_f.z;
    const float sc7 = scale8_f.w;

    const uint qs0_u32 = q_weight[block_word + 4u + q_offset / 4u];
    const uint qs64_u32 = q_weight[block_word + 4u + q_offset / 4u + 16u];
    const vec4 qs0_lo4 = unpack8_f32(qs0_u32 & 0x0F0F0F0Fu);
    const vec4 qs64_lo4 = unpack8_f32(qs64_u32 & 0x0F0F0F0Fu);
    const vec4 qs0_hi4 = unpack8_f32((qs0_u32 >> 4u) & 0x0F0F0F0Fu);
    const vec4 qs64_hi4 = unpack8_f32((qs64_u32 >> 4u) & 0x0F0F0F0Fu);

    const uint y1_idx = block_index * 256u + y_offset;
    const uint y2_idx = y1_idx + 128u;
    const vec4 by10 = load_x4(row, y1_idx);
    const vec4 by132 = load_x4(row, y1_idx + 32u);
    const vec4 by20 = load_x4(row, y2_idx);
    const vec4 by232 = load_x4(row, y2_idx + 32u);

    const float sx = dot(by10, qs0_lo4);
    const float sy = dot(by132, qs0_hi4);
    const float sz = dot(by20, qs64_lo4);
    const float sw = dot(by232, qs64_hi4);
    const float smin =
        dot(by10, vec4(sc2)) +
        dot(by132, vec4(sc3)) +
        dot(by20, vec4(sc6)) +
        dot(by232, vec4(sc7));
    return dm.x * (sx * sc0 + sy * sc1 + sz * sc4 + sw * sc5) - dm.y * smin;
}

float q4k_block_dot_k(
    uint row,
    uint col,
    uint block_index,
    uint blocks_per_row,
    uint q_offset,
    uint y_offset,
    uint v_im
) {
    const uint block_word = col * blocks_per_row * 36u + block_index * 36u;
    const vec2 dm = unpackHalf2x16(k_weight[block_word]);

    const uint scale0_u32 = q4k_u16_k(block_word, v_im);
    const uint scale4_u32 = q4k_u16_k(block_word, v_im + 2u);
    const uint scale8_u32 = q4k_u16_k(block_word, v_im + 4u);
    const uint scale_0_4_l = (scale4_u32 << 16u) | scale0_u32;
    const uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2u;
    const vec4 scale_0_4_l_f = unpack8_f32(scale_0_4_l & 0x3F3F3F3Fu);
    const vec4 scale8_f = unpack8_f32((((scale8_u32 << 12u) | scale8_u32) & 0x0F0F0F0Fu) | scale_0_4_h);

    const float sc0 = scale_0_4_l_f.x;
    const float sc1 = scale_0_4_l_f.y;
    const float sc2 = scale_0_4_l_f.z;
    const float sc3 = scale_0_4_l_f.w;
    const float sc4 = scale8_f.x;
    const float sc5 = scale8_f.y;
    const float sc6 = scale8_f.z;
    const float sc7 = scale8_f.w;

    const uint qs0_u32 = k_weight[block_word + 4u + q_offset / 4u];
    const uint qs64_u32 = k_weight[block_word + 4u + q_offset / 4u + 16u];
    const vec4 qs0_lo4 = unpack8_f32(qs0_u32 & 0x0F0F0F0Fu);
    const vec4 qs64_lo4 = unpack8_f32(qs64_u32 & 0x0F0F0F0Fu);
    const vec4 qs0_hi4 = unpack8_f32((qs0_u32 >> 4u) & 0x0F0F0F0Fu);
    const vec4 qs64_hi4 = unpack8_f32((qs64_u32 >> 4u) & 0x0F0F0F0Fu);

    const uint y1_idx = block_index * 256u + y_offset;
    const uint y2_idx = y1_idx + 128u;
    const vec4 by10 = load_x4(row, y1_idx);
    const vec4 by132 = load_x4(row, y1_idx + 32u);
    const vec4 by20 = load_x4(row, y2_idx);
    const vec4 by232 = load_x4(row, y2_idx + 32u);

    const float sx = dot(by10, qs0_lo4);
    const float sy = dot(by132, qs0_hi4);
    const float sz = dot(by20, qs64_lo4);
    const float sw = dot(by232, qs64_hi4);
    const float smin =
        dot(by10, vec4(sc2)) +
        dot(by132, vec4(sc3)) +
        dot(by20, vec4(sc6)) +
        dot(by232, vec4(sc7));
    return dm.x * (sx * sc0 + sy * sc1 + sz * sc4 + sw * sc5) - dm.y * smin;
}

float q4k_block_dot_v(
    uint row,
    uint col,
    uint block_index,
    uint blocks_per_row,
    uint q_offset,
    uint y_offset,
    uint v_im
) {
    const uint block_word = col * blocks_per_row * 36u + block_index * 36u;
    const vec2 dm = unpackHalf2x16(v_weight[block_word]);

    const uint scale0_u32 = q4k_u16_v(block_word, v_im);
    const uint scale4_u32 = q4k_u16_v(block_word, v_im + 2u);
    const uint scale8_u32 = q4k_u16_v(block_word, v_im + 4u);
    const uint scale_0_4_l = (scale4_u32 << 16u) | scale0_u32;
    const uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2u;
    const vec4 scale_0_4_l_f = unpack8_f32(scale_0_4_l & 0x3F3F3F3Fu);
    const vec4 scale8_f = unpack8_f32((((scale8_u32 << 12u) | scale8_u32) & 0x0F0F0F0Fu) | scale_0_4_h);

    const float sc0 = scale_0_4_l_f.x;
    const float sc1 = scale_0_4_l_f.y;
    const float sc2 = scale_0_4_l_f.z;
    const float sc3 = scale_0_4_l_f.w;
    const float sc4 = scale8_f.x;
    const float sc5 = scale8_f.y;
    const float sc6 = scale8_f.z;
    const float sc7 = scale8_f.w;

    const uint qs0_u32 = v_weight[block_word + 4u + q_offset / 4u];
    const uint qs64_u32 = v_weight[block_word + 4u + q_offset / 4u + 16u];
    const vec4 qs0_lo4 = unpack8_f32(qs0_u32 & 0x0F0F0F0Fu);
    const vec4 qs64_lo4 = unpack8_f32(qs64_u32 & 0x0F0F0F0Fu);
    const vec4 qs0_hi4 = unpack8_f32((qs0_u32 >> 4u) & 0x0F0F0F0Fu);
    const vec4 qs64_hi4 = unpack8_f32((qs64_u32 >> 4u) & 0x0F0F0F0Fu);

    const uint y1_idx = block_index * 256u + y_offset;
    const uint y2_idx = y1_idx + 128u;
    const vec4 by10 = load_x4(row, y1_idx);
    const vec4 by132 = load_x4(row, y1_idx + 32u);
    const vec4 by20 = load_x4(row, y2_idx);
    const vec4 by232 = load_x4(row, y2_idx + 32u);

    const float sx = dot(by10, qs0_lo4);
    const float sy = dot(by132, qs0_hi4);
    const float sz = dot(by20, qs64_lo4);
    const float sw = dot(by232, qs64_hi4);
    const float smin =
        dot(by10, vec4(sc2)) +
        dot(by132, vec4(sc3)) +
        dot(by20, vec4(sc6)) +
        dot(by232, vec4(sc7));
    return dm.x * (sx * sc0 + sy * sc1 + sz * sc4 + sw * sc5) - dm.y * smin;
}

float q4k_dot_q(uint row, uint col) {
    const uint blocks_per_row = pc.K / 256u;
    const uint lane = gl_SubgroupInvocationID;
    const uint itid = lane & 15u;
    const uint ix = lane >> 4u;
    const uint il = itid >> 2u;
    const uint ir = itid & 3u;
    const uint v_im = il >> 1u;
    const uint v_in = il & 1u;
    const uint l0 = 4u * (2u * ir + v_in);
    const uint q_offset = 32u * v_im + l0;
    const uint y_offset = 64u * v_im + l0;
    float acc = 0.0;
    for (uint block_index = ix; block_index < blocks_per_row; block_index += 4u) {
        acc += q4k_block_dot_q(row, col, block_index, blocks_per_row, q_offset, y_offset, v_im);
    }
    return subgroupAdd(acc);
}

float q4k_dot_k(uint row, uint col) {
    const uint blocks_per_row = pc.K / 256u;
    const uint lane = gl_SubgroupInvocationID;
    const uint itid = lane & 15u;
    const uint ix = lane >> 4u;
    const uint il = itid >> 2u;
    const uint ir = itid & 3u;
    const uint v_im = il >> 1u;
    const uint v_in = il & 1u;
    const uint l0 = 4u * (2u * ir + v_in);
    const uint q_offset = 32u * v_im + l0;
    const uint y_offset = 64u * v_im + l0;
    float acc = 0.0;
    for (uint block_index = ix; block_index < blocks_per_row; block_index += 4u) {
        acc += q4k_block_dot_k(row, col, block_index, blocks_per_row, q_offset, y_offset, v_im);
    }
    return subgroupAdd(acc);
}

float q4k_dot_v(uint row, uint col) {
    const uint blocks_per_row = pc.K / 256u;
    const uint lane = gl_SubgroupInvocationID;
    const uint itid = lane & 15u;
    const uint ix = lane >> 4u;
    const uint il = itid >> 2u;
    const uint ir = itid & 3u;
    const uint v_im = il >> 1u;
    const uint v_in = il & 1u;
    const uint l0 = 4u * (2u * ir + v_in);
    const uint q_offset = 32u * v_im + l0;
    const uint y_offset = 64u * v_im + l0;
    float acc = 0.0;
    for (uint block_index = ix; block_index < blocks_per_row; block_index += 4u) {
        acc += q4k_block_dot_v(row, col, block_index, blocks_per_row, q_offset, y_offset, v_im);
    }
    return subgroupAdd(acc);
}

void store_output(uint row, uint logical_col, float value) {
    if (logical_col < pc.NQ) {
        q_output[row * pc.NQ + logical_col] = float16_t(value);
        return;
    }
    if (logical_col < pc.NQ + pc.NK) {
        const uint k_col = logical_col - pc.NQ;
        k_output[row * pc.NK + k_col] = float16_t(value);
        return;
    }
    const uint v_col = logical_col - pc.NQ - pc.NK;
    if (v_col < pc.NV) {
        v_output[row * pc.NV + v_col] = float16_t(value);
    }
}

void main() {
    const uint lane = gl_SubgroupInvocationID;
    const uint row = gl_WorkGroupID.y;
    const uint total = total_n();
    const uint col0 = gl_WorkGroupID.x * 2u;
    const uint col1 = col0 + 1u;
    if (row >= pc.M) {
        return;
    }
    float acc0 = 0.0;
    float acc1 = 0.0;
    if (col0 < total) {
        if (col0 < pc.NQ) {
            acc0 = q4k_dot_q(row, col0);
        } else if (col0 < pc.NQ + pc.NK) {
            acc0 = q4k_dot_k(row, col0 - pc.NQ);
        } else {
            acc0 = q4k_dot_v(row, col0 - pc.NQ - pc.NK);
        }
    }
    if (col1 < total) {
        if (col1 < pc.NQ) {
            acc1 = q4k_dot_q(row, col1);
        } else if (col1 < pc.NQ + pc.NK) {
            acc1 = q4k_dot_k(row, col1 - pc.NQ);
        } else {
            acc1 = q4k_dot_v(row, col1 - pc.NQ - pc.NK);
        }
    }
    if (lane == 0u) {
        if (col0 < total) {
            store_output(row, col0, acc0);
        }
        if (col1 < total) {
            store_output(row, col1, acc1);
        }
    }
}
""",
)
