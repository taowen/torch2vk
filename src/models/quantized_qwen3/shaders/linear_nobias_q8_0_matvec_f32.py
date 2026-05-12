"""Generated shader: linear_nobias_q8_0_matvec_f32."""

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
    q8_0_halfwords_layout,
)


LINEAR_NOBIAS_Q8_0_MATVEC_F32 = ShaderVariant(
    name='linear_nobias_q8_0_matvec_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportLinearNobiasQ8_0MatvecProgram',
        shader_name='linear_nobias_q8_0_matvec_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('X0', 'X1', 'K',)),
            ),
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='uint16', shape=('N', mul(ceil_div('K', 32), 17),), layout=q8_0_halfwords_layout(logical_k='K', block_size=32, halfwords_per_block=17)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('X0', 'X1', 'N',)),
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
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

float q8_0_value(uint row, uint k) {
    const uint blocks_per_row = pc.K / 32u;
    const uint block_index = k >> 5u;
    const uint block_half = row * blocks_per_row * 17u + block_index * 17u;
    const float d = unpackHalf2x16(uint(weight[block_half])).x;
    const uint local = k & 31u;
    const uint packed = uint(weight[block_half + 1u + (local >> 1u)]);
    uint byte_value = ((local & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
    int quant = int(byte_value);
    if (quant >= 128) { quant -= 256; }
    return d * float(quant);
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
            const float x_value = float(x[row * pc.K + k]);
            if (col0 < pc.N) { acc0 = fma(x_value, q8_0_value(col0, k), acc0); }
            if (col1 < pc.N) { acc1 = fma(x_value, q8_0_value(col1, k), acc1); }
        }
    }
    acc0 = subgroupAdd(acc0);
    acc1 = subgroupAdd(acc1);
    if (lane == 0u && row < pc.M) {
        if (col0 < pc.N) { output_values[row * pc.N + col0] = float16_t(acc0); }
        if (col1 < pc.N) { output_values[row * pc.N + col1] = float16_t(acc1); }
    }
}
""",
)
