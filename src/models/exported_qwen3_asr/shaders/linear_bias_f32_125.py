"""Generated shader: linear_bias_f32_125."""

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


LINEAR_BIAS_F32_125 = ShaderVariant(
    name='linear_bias_f32_125',
    family='export',
    contract=ShaderContract(
        class_name='ExportLinearBiasProgram',
        shader_name='linear_bias_f32_125',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('X0', 'X1',)),
            ),
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='bfloat16', shape=('W0', 'W1',)),
            ),
            TensorFieldSpec(
                name='bias',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='bfloat16', shape=('N',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('Y0', 'Y1',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec('M', PushConstantType.UINT32, 0, 133, dynamic=False),
                PushConstantFieldSpec('K', PushConstantType.UINT32, 4, 896, dynamic=False),
                PushConstantFieldSpec('N', PushConstantType.UINT32, 8, 1024, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(133, 16), ceil_div(1024, 16), 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_bfloat16 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { bfloat16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { bfloat16_t bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
const uint TILE_M = 16u; const uint TILE_N = 16u; const uint TILE_K = 32u;
shared float tile_x[16 * 32]; shared bfloat16_t tile_w[32 * 16];
void main() {
    const uint local_col = gl_LocalInvocationID.x;
    const uint local_row = gl_LocalInvocationID.y;
    const uint lane = local_row * TILE_N + local_col;
    const uint row = gl_WorkGroupID.x * TILE_M + local_row;
    const uint col = gl_WorkGroupID.y * TILE_N + local_col;
    float acc = 0.0;
    for (uint k0 = 0u; k0 < pc.K; k0 += TILE_K) {
        for (uint i = lane; i < TILE_M * TILE_K; i += TILE_M * TILE_N) {
            const uint tr = i / TILE_K; const uint tk = i - tr * TILE_K;
            const uint gr = gl_WorkGroupID.x * TILE_M + tr; const uint gk = k0 + tk;
            tile_x[i] = (gr < pc.M && gk < pc.K) ? x[gr * pc.K + gk] : 0.0;
        }
        for (uint i = lane; i < TILE_K * TILE_N; i += TILE_M * TILE_N) {
            const uint tk = i / TILE_N; const uint tc = i - tk * TILE_N;
            const uint gk = k0 + tk; const uint gc = gl_WorkGroupID.y * TILE_N + tc;
            tile_w[i] = (gc < pc.N && gk < pc.K) ? weight[gc * pc.K + gk] : bfloat16_t(0.0);
        }
        barrier();
        [[unroll]] for (uint k = 0u; k < TILE_K; ++k) {
            acc = fma(tile_x[local_row * TILE_K + k], tile_w[k * TILE_N + local_col], acc);
        }
        barrier();
    }
    if (row < pc.M && col < pc.N) { output_values[row * pc.N + col] = fma(1.0, bias[col], acc); }
}
""",
)
