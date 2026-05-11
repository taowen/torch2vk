"""Generated shader: linear_nobias_q4_k_f32_40."""

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


LINEAR_NOBIAS_Q4_K_F32_40 = ShaderVariant(
    name='linear_nobias_q4_k_f32_40',
    family='export',
    contract=ShaderContract(
        class_name='ExportLinearNobiasQ4KProgram',
        shader_name='linear_nobias_q4_k_f32_40',
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
                PushConstantFieldSpec('M', PushConstantType.UINT32, 0, 170, dynamic=False),
                PushConstantFieldSpec('K', PushConstantType.UINT32, 4, 1024, dynamic=False),
                PushConstantFieldSpec('N', PushConstantType.UINT32, 8, 3072, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(170, 16), ceil_div(3072, 16), 1),
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
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint weight[]; };

layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const uint TILE_M = 16u;
const uint TILE_N = 16u;
const uint TILE_K = 16u;
const uint TILE_SIZE = TILE_M * TILE_K;

shared float16_t shared_a[TILE_SIZE];
shared float16_t shared_b[TILE_SIZE];
shared float shared_out[TILE_M * TILE_N];

shared float shared_q4_d[TILE_N];
shared float shared_q4_m[TILE_N];

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

uint q4k_quant(uint block_word, uint local_k) {
    const uint pair = local_k >> 6u;
    const uint byte_index = local_k & 31u;
    const uint packed_q = q4k_byte(block_word, 16u + pair * 32u + byte_index);
    return ((local_k & 32u) == 0u) ? (packed_q & 15u) : (packed_q >> 4u);
}

void prepare_q4k_tile_scales(uint lane, uint col_base, uint k_base) {
    const uint blocks_per_row = pc.K / 256u;
    const uint block_index = k_base >> 8u;
    const uint subblock = (k_base & 255u) >> 5u;
    for (uint col = lane; col < TILE_N; col += 64u) {
        const uint n = col_base + col;
        if (n < pc.N) {
            const uint block_word = n * blocks_per_row * 36u + block_index * 36u;
            const vec2 dm = unpackHalf2x16(weight[block_word]);
            uint scale;
            uint minimum;
            q4k_scale_min(block_word, subblock, scale, minimum);
            shared_q4_d[col] = dm.x * float(scale);
            shared_q4_m[col] = dm.y * float(minimum);
        } else {
            shared_q4_d[col] = 0.0;
            shared_q4_m[col] = 0.0;
        }
    }
}

void load_a_tile(uint lane, uint row_base, uint k_base) {
    for (uint i = lane; i < TILE_SIZE; i += 64u) {
        const uint row = i / TILE_K;
        const uint col = i - row * TILE_K;
        const uint m = row_base + row;
        const uint k = k_base + col;
        shared_a[i] = float16_t((m < pc.M && k < pc.K) ? x[m * pc.K + k] : 0.0);
    }
}

void load_b_tile_prepared(uint lane, uint col_base, uint k_base) {
    const uint blocks_per_row = pc.K / 256u;
    const uint block_index = k_base >> 8u;
    for (uint i = lane; i < TILE_SIZE; i += 64u) {
        const uint col = i / TILE_K;
        const uint k_offset = i - col * TILE_K;
        const uint n = col_base + col;
        const uint k = k_base + k_offset;
        if (n < pc.N && k < pc.K) {
            const uint block_word = n * blocks_per_row * 36u + block_index * 36u;
            const uint q = q4k_quant(block_word, k & 255u);
            shared_b[i] = float16_t(shared_q4_d[col] * float(q) - shared_q4_m[col]);
        } else {
            shared_b[i] = float16_t(0.0);
        }
    }
}

void main() {
    const uint lane = gl_SubgroupInvocationID;
    const uint row_base = gl_WorkGroupID.x * TILE_M;
    const uint col_base = gl_WorkGroupID.y * TILE_N;

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_a;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_b;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> mat_c;
    mat_c = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint k_base = 0u; k_base < pc.K; k_base += 32u) {
        prepare_q4k_tile_scales(lane, col_base, k_base);
        barrier();

        load_a_tile(lane, row_base, k_base);
        load_b_tile_prepared(lane, col_base, k_base);
        barrier();
        coopMatLoad(mat_a, shared_a, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(mat_b, shared_b, 0, int(TILE_K), gl_CooperativeMatrixLayoutColumnMajor);
        mat_c = coopMatMulAdd(mat_a, mat_b, mat_c);
        barrier();

        load_a_tile(lane, row_base, k_base + TILE_K);
        load_b_tile_prepared(lane, col_base, k_base + TILE_K);
        barrier();
        coopMatLoad(mat_a, shared_a, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(mat_b, shared_b, 0, int(TILE_K), gl_CooperativeMatrixLayoutColumnMajor);
        mat_c = coopMatMulAdd(mat_a, mat_b, mat_c);
        barrier();
    }

    coopMatStore(mat_c, shared_out, 0, int(TILE_N), gl_CooperativeMatrixLayoutRowMajor);
    barrier();

    for (uint i = lane; i < TILE_M * TILE_N; i += 64u) {
        const uint row = i / TILE_N;
        const uint col = i - row * TILE_N;
        const uint m = row_base + row;
        const uint n = col_base + col;
        if (m < pc.M && n < pc.N) {
            output_values[m * pc.N + n] = shared_out[i];
        }
    }
}
""",
)
