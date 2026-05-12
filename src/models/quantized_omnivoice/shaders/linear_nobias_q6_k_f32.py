"""Generated shader: linear_nobias_q6_k_f32."""

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
    CooperativeMatrixRequirements,
    SubgroupRequirements,
)
from torch2vk.vulkan.types import (
    q6_k_halfwords_layout,
)


LINEAR_NOBIAS_Q6_K_F32 = ShaderVariant(
    name='linear_nobias_q6_k_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportLinearNobiasQ6KProgram',
        shader_name='linear_nobias_q6_k_f32',
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
                contract=TensorContract(dtype='uint16', shape=('N', mul(ceil_div('K', 256), 105),), layout=q6_k_halfwords_layout(logical_k='K', block_size=256, halfwords_per_block=105)),
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
        dispatch=(ceil_div('N', 64), ceil_div(mul('X0', 'X1'), 64), 1),
    ),
    execution_requirements=ShaderExecutionRequirements(subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True), cooperative_matrix=CooperativeMatrixRequirements(scope='subgroup', m_size=16, n_size=16, k_size=16, a_type='float16', b_type='float16', c_type='float16', result_type='float16', saturating_accumulation=False), require_storage_buffer_16bit_access=True),
    source="""\
#version 460

#pragma use_vulkan_memory_model

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_cooperative_matrix : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 0) buffer restrict readonly XVec4Buffer { f16vec4 x4[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

const uint TILE_N = 64u;
const uint TILE_M = 64u;
const uint TILE_K = 32u;
const uint TM = 16u;
const uint TN = 16u;
const uint TK = 16u;
const uint WARP = 64u;
const uint WARPS = 2u;
const uint WN = 32u;
const uint CMS_PER_ROW = TILE_N / TM;
const uint CMS_PER_COL = WN / TN;
const uint TILE_NK = TILE_N * TILE_K;
const uint TILE_MK = TILE_M * TILE_K;
const uint WEIGHT_VALUES_PER_LOAD = 4u;
const uint X_VALUES_PER_LOAD = 8u;
const uint TILE_K_WEIGHT_LOADS = TILE_K / WEIGHT_VALUES_PER_LOAD;
const uint TILE_K_X_LOADS = TILE_K / X_VALUES_PER_LOAD;
const uint TILE_NK_LOADS = TILE_N * TILE_K_WEIGHT_LOADS;
const uint TILE_MK_LOADS = TILE_M * TILE_K_X_LOADS;
const uint STAGE_SIZE = TM * TN;
const uint SHMEM_STRIDE = TILE_K / 2u + 4u;

shared f16vec2 shared_w[TILE_N * SHMEM_STRIDE];
shared f16vec2 shared_x[TILE_M * SHMEM_STRIDE];
shared float16_t shared_stage[WARPS * STAGE_SIZE];

uint q6k_byte(uint block_half, uint byte_offset) {
    const uint packed = uint(weight[block_half + (byte_offset >> 1u)]);
    return ((byte_offset & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
}

int q6k_i8(uint block_half, uint byte_offset) {
    int value = int(q6k_byte(block_half, byte_offset));
    if (value >= 128) {
        value -= 256;
    }
    return value;
}

f16vec2 q6k_pair(uint block_half, uint local_k) {
    const uint iqs = (local_k >> 1u) & 127u;
    const uint section = iqs >> 6u;
    const uint b = ((iqs & 63u) >> 5u) * 4u;
    const uint is_b = (iqs & 15u) >> 3u;
    const uint qhshift = ((iqs & 63u) >> 4u) * 2u;
    const uint scale_index = 8u * section + qhshift + is_b;
    const uint qsi = section * 32u + (iqs & 31u);
    const uint qhi = section * 16u + (iqs & 15u);
    const float d = unpackHalf2x16(uint(weight[block_half + 104u])).x;
    const float dscale = d * float(q6k_i8(block_half, 192u + scale_index));
    const uint ql = (uint(weight[block_half + qsi]) >> b) & 0x0F0Fu;
    const uint qh = (uint(weight[block_half + 64u + qhi]) >> qhshift) & 0x0303u;
    const uint packed = ql | (qh << 4u);
    return f16vec2(
        float16_t((float(int(packed & 255u)) - 32.0) * dscale),
        float16_t((float(int((packed >> 8u) & 255u)) - 32.0) * dscale)
    );
}

void load_weight_tile(uint local_id, uint n_base, uint k_base) {
    const uint blocks_per_row = pc.K / 256u;
    const uint block_index = k_base >> 8u;
    for (uint i = local_id; i < TILE_NK_LOADS; i += 128u) {
        const uint row = i / TILE_K_WEIGHT_LOADS;
        const uint k_offset = (i - row * TILE_K_WEIGHT_LOADS) * WEIGHT_VALUES_PER_LOAD;
        const uint n = n_base + row;
        const uint k = k_base + k_offset;
        const uint base = row * SHMEM_STRIDE + k_offset / 2u;
        if (n < pc.N && k < pc.K) {
            const uint block_half = n * blocks_per_row * 105u + block_index * 105u;
            const uint local_k = k & 255u;
            shared_w[base] = q6k_pair(block_half, local_k);
            shared_w[base + 1u] = q6k_pair(block_half, local_k + 2u);
        } else {
            shared_w[base] = f16vec2(0.0);
            shared_w[base + 1u] = f16vec2(0.0);
        }
    }
}

void load_x_tile(uint local_id, uint m_base, uint k_base) {
    for (uint i = local_id; i < TILE_MK_LOADS; i += 128u) {
        const uint row = i / TILE_K_X_LOADS;
        const uint k_offset = (i - row * TILE_K_X_LOADS) * X_VALUES_PER_LOAD;
        const uint m = m_base + row;
        const uint k = k_base + k_offset;
        const uint base = row * SHMEM_STRIDE + k_offset / 2u;
        if (m < pc.M && k + 7u < pc.K) {
            const uint input_base = m * pc.K + k;
            const f16vec4 values0 = x4[input_base >> 2u];
            const f16vec4 values1 = x4[(input_base >> 2u) + 1u];
            shared_x[base] = f16vec2(values0.x, values0.y);
            shared_x[base + 1u] = f16vec2(values0.z, values0.w);
            shared_x[base + 2u] = f16vec2(values1.x, values1.y);
            shared_x[base + 3u] = f16vec2(values1.z, values1.w);
        } else {
            shared_x[base] = f16vec2(
                float16_t((m < pc.M && k < pc.K) ? float(x[m * pc.K + k]) : 0.0),
                float16_t((m < pc.M && k + 1u < pc.K) ? float(x[m * pc.K + k + 1u]) : 0.0)
            );
            shared_x[base + 1u] = f16vec2(
                float16_t((m < pc.M && k + 2u < pc.K) ? float(x[m * pc.K + k + 2u]) : 0.0),
                float16_t((m < pc.M && k + 3u < pc.K) ? float(x[m * pc.K + k + 3u]) : 0.0)
            );
            shared_x[base + 2u] = f16vec2(
                float16_t((m < pc.M && k + 4u < pc.K) ? float(x[m * pc.K + k + 4u]) : 0.0),
                float16_t((m < pc.M && k + 5u < pc.K) ? float(x[m * pc.K + k + 5u]) : 0.0)
            );
            shared_x[base + 3u] = f16vec2(
                float16_t((m < pc.M && k + 6u < pc.K) ? float(x[m * pc.K + k + 6u]) : 0.0),
                float16_t((m < pc.M && k + 7u < pc.K) ? float(x[m * pc.K + k + 7u]) : 0.0)
            );
        }
    }
}

void main() {
    const uint local_id = gl_LocalInvocationID.x;
    const uint lane = gl_SubgroupInvocationID;
    const uint warp = gl_SubgroupID;
    const uint n_base = gl_WorkGroupID.x * TILE_N;
    const uint m_base = gl_WorkGroupID.y * TILE_M;

    const uint store_r = lane % TM;
    const uint store_c = lane / TM;
    const uint store_stride = WARP / TM;

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_w;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_x;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> sums[CMS_PER_ROW * CMS_PER_COL];

    [[unroll]] for (uint i = 0u; i < CMS_PER_ROW * CMS_PER_COL; ++i) {
        sums[i] = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(float16_t(0.0));
    }

    for (uint k_base = 0u; k_base < pc.K; k_base += TILE_K) {
        load_weight_tile(local_id, n_base, k_base);
        load_x_tile(local_id, m_base, k_base);
        barrier();

        [[unroll]] for (uint kk = 0u; kk < TILE_K; kk += TK) {
            [[unroll]] for (uint cm_row = 0u; cm_row < CMS_PER_ROW; ++cm_row) {
                coopMatLoad(mat_w, shared_w, int(cm_row * TM * SHMEM_STRIDE + kk / 2u), int(SHMEM_STRIDE), gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint cm_col = 0u; cm_col < CMS_PER_COL; ++cm_col) {
                    coopMatLoad(mat_x, shared_x, int((warp * WN + cm_col * TN) * SHMEM_STRIDE + kk / 2u), int(SHMEM_STRIDE), gl_CooperativeMatrixLayoutColumnMajor);
                    const uint sum_index = cm_col * CMS_PER_ROW + cm_row;
                    sums[sum_index] = coopMatMulAdd(mat_w, mat_x, sums[sum_index]);
                }
            }
        }
        barrier();
    }

    [[unroll]] for (uint cm_row = 0u; cm_row < CMS_PER_ROW; ++cm_row) {
        [[unroll]] for (uint cm_col = 0u; cm_col < CMS_PER_COL; ++cm_col) {
            const uint sum_index = cm_col * CMS_PER_ROW + cm_row;
            const uint n_tile = n_base + cm_row * TM;
            const uint m_tile = m_base + warp * WN + cm_col * TN;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> output_tile =
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(sums[sum_index]);
            if (n_tile + TM <= pc.N && m_tile + TN <= pc.M) {
                coopMatStore(output_tile, output_values, int(m_tile * pc.N + n_tile), int(pc.N), gl_CooperativeMatrixLayoutColumnMajor);
            } else {
                coopMatStore(output_tile, shared_stage, int(warp * STAGE_SIZE), int(TM), gl_CooperativeMatrixLayoutColumnMajor);
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);

                [[unroll]] for (uint col = 0u; col < TN; col += store_stride) {
                    const uint n = n_tile + store_r;
                    const uint m = m_tile + col + store_c;
                    if (n < pc.N && m < pc.M) {
                        const uint stage_index = warp * STAGE_SIZE + (col + store_c) * TM + store_r;
                        output_values[m * pc.N + n] = shared_stage[stage_index];
                    }
                }
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
            }
        }
    }
}
""",
)
