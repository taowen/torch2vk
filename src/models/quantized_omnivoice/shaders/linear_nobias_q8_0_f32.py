"""Generated shader: linear_nobias_q8_0_f32."""

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


LINEAR_NOBIAS_Q8_0_F32 = ShaderVariant(
    name='linear_nobias_q8_0_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportLinearNobiasQ8_0Program',
        shader_name='linear_nobias_q8_0_f32',
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
                contract=TensorContract(dtype='uint16', shape=('N', mul(ceil_div('K', 32), 17),), layout=q8_0_halfwords_layout(logical_k='K', block_size=32, halfwords_per_block=17)),
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
                PushConstantFieldSpec('M', PushConstantType.UINT32, 0, 170, dynamic=False),
                PushConstantFieldSpec('K', PushConstantType.UINT32, 4, 1024, dynamic=False),
                PushConstantFieldSpec('N', PushConstantType.UINT32, 8, 8200, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(170, 32), ceil_div(8200, 16), 1),
    ),
    execution_requirements=ShaderExecutionRequirements(subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True), require_storage_buffer_16bit_access=True),
    source="""\
#version 460

#pragma use_vulkan_memory_model

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_cooperative_matrix : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };

layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const uint TILE_M = 16u;
const uint GROUP_M = 32u;
const uint TILE_N = 16u;
const uint TILE_K = 16u;
const uint TILE_SIZE = TILE_M * TILE_K;
const uint OUT_SIZE = TILE_M * TILE_N;

shared float16_t shared_a0[TILE_SIZE];
shared float16_t shared_a1[TILE_SIZE];
shared float16_t shared_b[TILE_SIZE];
shared float shared_out0[OUT_SIZE];
shared float shared_out1[OUT_SIZE];

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
void load_a_tile_pair(uint lane, uint row_base, uint k_base) {
    for (uint i = lane; i < TILE_SIZE; i += 64u) {
        const uint row = i / TILE_K;
        const uint col = i - row * TILE_K;
        const uint m0 = row_base + row;
        const uint m1 = row_base + TILE_M + row;
        const uint k = k_base + col;
        shared_a0[i] = float16_t((m0 < pc.M && k < pc.K) ? x[m0 * pc.K + k] : 0.0);
        shared_a1[i] = float16_t((m1 < pc.M && k < pc.K) ? x[m1 * pc.K + k] : 0.0);
    }
}

void load_b_tile(uint lane, uint col_base, uint k_base) {
    for (uint i = lane; i < TILE_SIZE; i += 64u) {
        const uint col = i / TILE_K;
        const uint k_offset = i - col * TILE_K;
        const uint n = col_base + col;
        const uint k = k_base + k_offset;
        shared_b[i] = float16_t((n < pc.N && k < pc.K) ? q8_0_value(n, k) : 0.0);
    }
}

void main() {
    const uint lane = gl_SubgroupInvocationID;
    const uint row_base = gl_WorkGroupID.x * GROUP_M;
    const uint col_base = gl_WorkGroupID.y * TILE_N;

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_a0;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_a1;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_b;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> mat_c0;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> mat_c1;
    mat_c0 = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    mat_c1 = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint k_base = 0u; k_base < pc.K; k_base += TILE_K) {
        load_a_tile_pair(lane, row_base, k_base);
        load_b_tile(lane, col_base, k_base);
        barrier();
        coopMatLoad(mat_a0, shared_a0, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(mat_a1, shared_a1, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(mat_b, shared_b, 0, int(TILE_K), gl_CooperativeMatrixLayoutColumnMajor);
        mat_c0 = coopMatMulAdd(mat_a0, mat_b, mat_c0);
        mat_c1 = coopMatMulAdd(mat_a1, mat_b, mat_c1);
        barrier();
    }

    coopMatStore(mat_c0, shared_out0, 0, int(TILE_N), gl_CooperativeMatrixLayoutRowMajor);
    coopMatStore(mat_c1, shared_out1, 0, int(TILE_N), gl_CooperativeMatrixLayoutRowMajor);
    barrier();

    for (uint i = lane; i < OUT_SIZE; i += 64u) {
        const uint row = i / TILE_N;
        const uint col = i - row * TILE_N;
        const uint m0 = row_base + row;
        const uint m1 = row_base + TILE_M + row;
        const uint n = col_base + col;
        if (n < pc.N) {
            if (m0 < pc.M) {
                output_values[m0 * pc.N + n] = shared_out0[i];
            }
            if (m1 < pc.M) {
                output_values[m1 * pc.N + n] = shared_out1[i];
            }
        }
    }
}
""",
)
