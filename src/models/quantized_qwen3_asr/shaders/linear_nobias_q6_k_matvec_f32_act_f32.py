"""Generated shader: linear_nobias_q6_k_matvec_f32_act_f32."""

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
    q6_k_halfwords_layout,
)


LINEAR_NOBIAS_Q6_K_MATVEC_F32_ACT_F32 = ShaderVariant(
    name='linear_nobias_q6_k_matvec_f32_act_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportLinearNobiasQ6KMatvecProgram',
        shader_name='linear_nobias_q6_k_matvec_f32_act_f32',
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
                contract=TensorContract(dtype='uint16', shape=('N', mul(ceil_div('K', 256), 105),), layout=q6_k_halfwords_layout(logical_k='K', block_size=256, halfwords_per_block=105)),
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
    execution_requirements=ShaderExecutionRequirements(subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True), require_storage_buffer_16bit_access=True),
    source="""\
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 0) buffer restrict readonly XVec4Buffer { vec4 x4[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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

vec4 load_x4(uint row, uint k) {
    return vec4(x4[(row * pc.K + k) >> 2u]);
}

float q6k_block_dot(uint row, uint col, uint block_index, uint itid) {
    const uint blocks_per_row = pc.K / 256u;
    const uint block_half = col * blocks_per_row * 105u + block_index * 105u;
    const uint k_base = block_index * 256u;

    const uint v_im = itid >> 3u;
    const uint v_in = itid - 8u * v_im;
    const uint l0 = 4u * v_in;
    const uint is = v_in >> 2u;
    const uint x_offset = k_base + 128u * v_im + l0;

    const uint ql_offset = 64u * v_im + l0;
    const uint qh_offset = 128u + 32u * v_im + l0;
    const uint scale_offset = 192u + 8u * v_im + is;

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

    const vec4 x0 = load_x4(row, x_offset);
    const vec4 x1 = load_x4(row, x_offset + 32u);
    const vec4 x2 = load_x4(row, x_offset + 64u);
    const vec4 x3 = load_x4(row, x_offset + 96u);

    const float d = unpackHalf2x16(uint(weight[block_half + 104u])).x;
    const float s0 = float(q6k_i8(block_half, scale_offset));
    const float s1 = float(q6k_i8(block_half, scale_offset + 2u));
    const float s2 = float(q6k_i8(block_half, scale_offset + 4u));
    const float s3 = float(q6k_i8(block_half, scale_offset + 6u));
    return d * (dot(q0, x0) * s0 + dot(q1, x1) * s1 + dot(q2, x2) * s2 + dot(q3, x3) * s3);
}

void main() {
    const uint col0 = gl_WorkGroupID.x * 2u;
    const uint col1 = col0 + 1u;
    const uint row = gl_WorkGroupID.y;
    const uint lane = gl_SubgroupInvocationID;
    if (row >= pc.M) {
        return;
    }

    const uint itid = lane & 15u;
    const uint ix = lane >> 4u;
    const uint blocks_per_row = pc.K / 256u;
    float acc0 = 0.0;
    float acc1 = 0.0;
    for (uint block = ix; block < blocks_per_row; block += 4u) {
        if (col0 < pc.N) {
            acc0 += q6k_block_dot(row, col0, block, itid);
        }
        if (col1 < pc.N) {
            acc1 += q6k_block_dot(row, col1, block, itid);
        }
    }

    acc0 = subgroupAdd(acc0);
    acc1 = subgroupAdd(acc1);
    if (lane == 0u) {
        if (col0 < pc.N) {
            output_values[row * pc.N + col0] = acc0;
        }
        if (col1 < pc.N) {
            output_values[row * pc.N + col1] = acc1;
        }
    }
}
""",
)
