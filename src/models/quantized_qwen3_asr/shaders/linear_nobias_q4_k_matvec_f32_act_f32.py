"""Generated shader: linear_nobias_q4_k_matvec_f32_act_f32."""

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
from torch2vk.vulkan.types import (
    q4_k_words_layout,
)


LINEAR_NOBIAS_Q4_K_MATVEC_F32_ACT_F32 = ShaderVariant(
    name='linear_nobias_q4_k_matvec_f32_act_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportLinearNobiasQ4KMatvecProgram',
        shader_name='linear_nobias_q4_k_matvec_f32_act_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('X0', 'X1', 'K',)),
            ),
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='uint32', shape=('N', mul(ceil_div('K', 256), 36),), layout=q4_k_words_layout(logical_k='K', block_size=256, words_per_block=36)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('X0', 'X1', 'N',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec('M', PushConstantType.UINT32, 0, mul('X0', 'X1'), dynamic=False),
                PushConstantFieldSpec('K', PushConstantType.UINT32, 4, 'K', dynamic=False),
                PushConstantFieldSpec('N', PushConstantType.UINT32, 8, 'N', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div('N', 2), mul('X0', 'X1'), 1),
    ),
    execution_requirements=ShaderExecutionRequirements(subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True)),
    source="""\
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 0) buffer restrict readonly XVec4Buffer { vec4 x4[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

uint q4k_byte(uint block_word, uint byte_offset) {
    const uint word_value = weight[block_word + (byte_offset >> 2u)];
    return (word_value >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

uint q4k_u16(uint block_word, uint half_index) {
    const uint byte_offset = 4u + half_index * 2u;
    return q4k_byte(block_word, byte_offset) | (q4k_byte(block_word, byte_offset + 1u) << 8u);
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

float q4k_block_dot(
    uint row,
    uint col,
    uint block_index,
    uint blocks_per_row,
    uint q_offset,
    uint y_offset,
    uint v_im
) {
    const uint block_word = col * blocks_per_row * 36u + block_index * 36u;
    const vec2 dm = unpackHalf2x16(weight[block_word]);

    const uint scale0_u32 = q4k_u16(block_word, v_im);
    const uint scale4_u32 = q4k_u16(block_word, v_im + 2u);
    const uint scale8_u32 = q4k_u16(block_word, v_im + 4u);
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

    const uint qs0_u32 = weight[block_word + 4u + q_offset / 4u];
    const uint qs64_u32 = weight[block_word + 4u + q_offset / 4u + 16u];
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

void main() {
    const uint lane = gl_SubgroupInvocationID;
    const uint row = gl_WorkGroupID.y;
    const uint col0 = gl_WorkGroupID.x * 2u;
    const uint col1 = col0 + 1u;
    float acc0 = 0.0;
    float acc1 = 0.0;
    if (row < pc.M) {
        const uint blocks_per_row = pc.K / 256u;
        const uint itid = lane & 15u;
        const uint ix = lane >> 4u;
        const uint il = itid >> 2u;
        const uint ir = itid & 3u;
        const uint v_im = il >> 1u;
        const uint v_in = il & 1u;
        const uint l0 = 4u * (2u * ir + v_in);
        const uint q_offset = 32u * v_im + l0;
        const uint y_offset = 64u * v_im + l0;
        for (uint block_index = ix; block_index < blocks_per_row; block_index += 4u) {
            if (col0 < pc.N) {
                acc0 += q4k_block_dot(row, col0, block_index, blocks_per_row, q_offset, y_offset, v_im);
            }
            if (col1 < pc.N) {
                acc1 += q4k_block_dot(row, col1, block_index, blocks_per_row, q_offset, y_offset, v_im);
            }
        }
    }
    acc0 = subgroupAdd(acc0);
    acc1 = subgroupAdd(acc1);
    if (lane == 0u && row < pc.M) {
        if (col0 < pc.N) { output_values[row * pc.N + col0] = acc0; }
        if (col1 < pc.N) { output_values[row * pc.N + col1] = acc1; }
    }
}
""",
)
