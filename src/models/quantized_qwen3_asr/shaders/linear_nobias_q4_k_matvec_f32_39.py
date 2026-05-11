"""Generated shader: linear_nobias_q4_k_matvec_f32_39."""

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


LINEAR_NOBIAS_Q4_K_MATVEC_F32_39 = ShaderVariant(
    name='linear_nobias_q4_k_matvec_f32_39',
    family='export',
    contract=ShaderContract(
        class_name='ExportLinearNobiasQ4KMatvecProgram',
        shader_name='linear_nobias_q4_k_matvec_f32_39',
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
                PushConstantFieldSpec('M', PushConstantType.UINT32, 0, 1, dynamic=False),
                PushConstantFieldSpec('K', PushConstantType.UINT32, 4, 1024, dynamic=False),
                PushConstantFieldSpec('N', PushConstantType.UINT32, 8, 3072, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(3072, 2), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True)),
    source="""\
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

uint q4k_byte(uint block_word, uint byte_offset) {
    const uint word_value = weight[block_word + (byte_offset >> 2u)];
    return (word_value >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

void q4k_scale_min(uint block_word, uint subblock, out uint scale, out uint minimum) {
    if (subblock < 4u) {
        scale = q4k_byte(block_word, 4u + subblock) & 63u;
        minimum = q4k_byte(block_word, 8u + subblock) & 63u;
        return;
    }
    const uint local = subblock - 4u;
    const uint d_byte = q4k_byte(block_word, 4u + local);
    const uint m_byte = q4k_byte(block_word, 8u + local);
    const uint packed = q4k_byte(block_word, 12u + local);
    scale = (packed & 15u) | ((d_byte >> 2u) & 48u);
    minimum = (packed >> 4u) | ((m_byte >> 2u) & 48u);
}

float q4k_value(uint row, uint k) {
    const uint blocks_per_row = pc.K / 256u;
    const uint block_index = k >> 8u;
    const uint block_word = row * blocks_per_row * 36u + block_index * 36u;
    const vec2 dm = unpackHalf2x16(weight[block_word]);
    const uint local_k = k & 255u;
    const uint subblock = local_k >> 5u;
    uint scale;
    uint minimum;
    q4k_scale_min(block_word, subblock, scale, minimum);
    const uint pair = local_k >> 6u;
    const uint byte_index = local_k & 31u;
    const uint packed_q = q4k_byte(block_word, 16u + pair * 32u + byte_index);
    const uint q = ((local_k & 32u) == 0u) ? (packed_q & 15u) : (packed_q >> 4u);
    return dm.x * float(scale) * float(q) - dm.y * float(minimum);
}

void main() {
    const uint lane = gl_SubgroupInvocationID;
    const uint row = gl_WorkGroupID.y;
    const uint col0 = gl_WorkGroupID.x * 2u;
    const uint col1 = col0 + 1u;
    float acc0 = 0.0;
    float acc1 = 0.0;
    if (row < pc.M) {
        for (uint k = lane; k < pc.K; k += 64u) {
            const float x_value = x[row * pc.K + k];
            if (col0 < pc.N) { acc0 = fma(x_value, q4k_value(col0, k), acc0); }
            if (col1 < pc.N) { acc1 = fma(x_value, q4k_value(col1, k), acc1); }
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
