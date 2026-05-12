"""Fused Q4_K cooperative-matrix gate/up projection with SiLU."""

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
    CooperativeMatrixRequirements,
    ShaderExecutionRequirements,
    SubgroupRequirements,
)
from torch2vk.vulkan.types import q4_k_words_layout


LINEAR_NOBIAS_Q4_K_SWIGLU_F32 = ShaderVariant(
    name="linear_nobias_q4_k_swiglu_f32",
    family="optimized_omnivoice",
    contract=ShaderContract(
        class_name="LinearNobiasQ4KSwiGLUProgram",
        shader_name="linear_nobias_q4_k_swiglu_f32",
        fields=(
            TensorFieldSpec(
                "x", IOKind.INPUT, "input", TensorContract(dtype="float16", shape=("X0", "X1", "K"))
            ),
            TensorFieldSpec(
                "gate_weight",
                IOKind.INPUT,
                "weight",
                TensorContract(
                    dtype="uint32",
                    shape=("N", mul(ceil_div("K", 256), 36)),
                    layout=q4_k_words_layout(logical_k="K"),
                ),
            ),
            TensorFieldSpec(
                "up_weight",
                IOKind.INPUT,
                "weight",
                TensorContract(
                    dtype="uint32",
                    shape=("N", mul(ceil_div("K", 256), 36)),
                    layout=q4_k_words_layout(logical_k="K"),
                ),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float16", shape=("X0", "X1", "N")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("M", PushConstantType.UINT32, 0, mul("X0", "X1")),
                PushConstantFieldSpec("K", PushConstantType.UINT32, 4, "K"),
                PushConstantFieldSpec("N", PushConstantType.UINT32, 8, "N"),
            ),
        ),
        dispatch=(ceil_div("N", 64), ceil_div(mul("X0", "X1"), 64), 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        cooperative_matrix=CooperativeMatrixRequirements(
            scope="subgroup",
            m_size=16,
            n_size=16,
            k_size=16,
            a_type="float16",
            b_type="float16",
            c_type="float16",
            result_type="float16",
        ),
        require_storage_buffer_16bit_access=True,
    ),
    source="""\
#version 460

#pragma use_vulkan_memory_model

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_cooperative_matrix : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 0) buffer restrict readonly XVec4Buffer { f16vec4 x4[]; };
layout(set = 0, binding = 1) buffer restrict readonly GateWeightBuffer { uint gate_weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly UpWeightBuffer { uint up_weight[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };

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

shared f16vec2 shared_gate_w[TILE_N * SHMEM_STRIDE];
shared f16vec2 shared_up_w[TILE_N * SHMEM_STRIDE];
shared f16vec2 shared_x[TILE_M * SHMEM_STRIDE];
shared float16_t shared_gate_stage[WARPS * STAGE_SIZE];
shared float16_t shared_up_stage[WARPS * STAGE_SIZE];
shared float shared_gate_d[TILE_N];
shared float shared_gate_m[TILE_N];
shared float shared_up_d[TILE_N];
shared float shared_up_m[TILE_N];

uint weight_word(bool up, uint index) {
    return up ? up_weight[index] : gate_weight[index];
}

uint q4k_byte(bool up, uint block_word, uint byte_offset) {
    const uint word_value = weight_word(up, block_word + (byte_offset >> 2u));
    return (word_value >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

void q4k_scale_min(bool up, uint block_word, uint subblock, out uint scale, out uint minimum) {
    if (subblock < 4u) {
        scale = q4k_byte(up, block_word, 4u + subblock) & 63u;
        minimum = q4k_byte(up, block_word, 8u + subblock) & 63u;
        return;
    }
    const uint local = subblock - 4u;
    const uint d_byte = q4k_byte(up, block_word, 4u + local);
    const uint m_byte = q4k_byte(up, block_word, 8u + local);
    const uint packed = q4k_byte(up, block_word, 12u + local);
    scale = (packed & 15u) | ((d_byte >> 2u) & 48u);
    minimum = (packed >> 4u) | ((m_byte >> 2u) & 48u);
}

uint q4k_quant4(bool up, uint block_word, uint local_k) {
    const uint pair = local_k >> 6u;
    const uint byte_index = local_k & 31u;
    const uint packed_q = weight_word(up, block_word + ((16u + pair * 32u + byte_index) >> 2u));
    return ((local_k & 32u) == 0u) ? (packed_q & 0x0F0F0F0Fu) : ((packed_q >> 4u) & 0x0F0F0F0Fu);
}

void prepare_q4k_tile_scales(uint local_id, uint n_base, uint k_base) {
    const uint blocks_per_row = pc.K / 256u;
    const uint block_index = k_base >> 8u;
    const uint subblock = (k_base & 255u) >> 5u;
    for (uint row = local_id; row < TILE_N; row += 128u) {
        const uint n = n_base + row;
        if (n < pc.N) {
            const uint block_word = n * blocks_per_row * 36u + block_index * 36u;
            uint gate_scale;
            uint gate_minimum;
            const vec2 gate_dm = unpackHalf2x16(gate_weight[block_word]);
            q4k_scale_min(false, block_word, subblock, gate_scale, gate_minimum);
            shared_gate_d[row] = gate_dm.x * float(gate_scale);
            shared_gate_m[row] = gate_dm.y * float(gate_minimum);

            uint up_scale;
            uint up_minimum;
            const vec2 up_dm = unpackHalf2x16(up_weight[block_word]);
            q4k_scale_min(true, block_word, subblock, up_scale, up_minimum);
            shared_up_d[row] = up_dm.x * float(up_scale);
            shared_up_m[row] = up_dm.y * float(up_minimum);
        } else {
            shared_gate_d[row] = 0.0;
            shared_gate_m[row] = 0.0;
            shared_up_d[row] = 0.0;
            shared_up_m[row] = 0.0;
        }
    }
}

void load_weight_tile(uint local_id, uint n_base, uint k_base) {
    const uint blocks_per_row = pc.K / 256u;
    const uint block_index = k_base >> 8u;
    for (uint i = local_id; i < TILE_NK_LOADS; i += 128u) {
        const uint row = i / TILE_K_WEIGHT_LOADS;
        const uint k_offset = (i - row * TILE_K_WEIGHT_LOADS) * WEIGHT_VALUES_PER_LOAD;
        const uint n = n_base + row;
        const uint base = row * SHMEM_STRIDE + k_offset / 2u;
        if (n < pc.N && k_base + k_offset < pc.K) {
            const uint block_word = n * blocks_per_row * 36u + block_index * 36u;
            const uint local_k = (k_base + k_offset) & 255u;
            const uint gate_q = q4k_quant4(false, block_word, local_k);
            const uint up_q = q4k_quant4(true, block_word, local_k);
            const uint gate_q0 = gate_q & 15u;
            const uint gate_q1 = (gate_q >> 8u) & 15u;
            const uint gate_q2 = (gate_q >> 16u) & 15u;
            const uint gate_q3 = (gate_q >> 24u) & 15u;
            const uint up_q0 = up_q & 15u;
            const uint up_q1 = (up_q >> 8u) & 15u;
            const uint up_q2 = (up_q >> 16u) & 15u;
            const uint up_q3 = (up_q >> 24u) & 15u;
            shared_gate_w[base] = f16vec2(
                float16_t(shared_gate_d[row] * float(gate_q0) - shared_gate_m[row]),
                float16_t(shared_gate_d[row] * float(gate_q1) - shared_gate_m[row])
            );
            shared_gate_w[base + 1u] = f16vec2(
                float16_t(shared_gate_d[row] * float(gate_q2) - shared_gate_m[row]),
                float16_t(shared_gate_d[row] * float(gate_q3) - shared_gate_m[row])
            );
            shared_up_w[base] = f16vec2(
                float16_t(shared_up_d[row] * float(up_q0) - shared_up_m[row]),
                float16_t(shared_up_d[row] * float(up_q1) - shared_up_m[row])
            );
            shared_up_w[base + 1u] = f16vec2(
                float16_t(shared_up_d[row] * float(up_q2) - shared_up_m[row]),
                float16_t(shared_up_d[row] * float(up_q3) - shared_up_m[row])
            );
        } else {
            shared_gate_w[base] = f16vec2(0.0);
            shared_gate_w[base + 1u] = f16vec2(0.0);
            shared_up_w[base] = f16vec2(0.0);
            shared_up_w[base + 1u] = f16vec2(0.0);
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

float swiglu_value(float gate, float up) {
    return (gate / (1.0 + exp(-gate))) * up;
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

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> gate_w;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> up_w;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_x;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> gate_sums[CMS_PER_ROW * CMS_PER_COL];
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> up_sums[CMS_PER_ROW * CMS_PER_COL];

    [[unroll]] for (uint i = 0u; i < CMS_PER_ROW * CMS_PER_COL; ++i) {
        gate_sums[i] = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(float16_t(0.0));
        up_sums[i] = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(float16_t(0.0));
    }

    for (uint k_base = 0u; k_base < pc.K; k_base += TILE_K) {
        prepare_q4k_tile_scales(local_id, n_base, k_base);
        barrier();
        load_weight_tile(local_id, n_base, k_base);
        load_x_tile(local_id, m_base, k_base);
        barrier();

        [[unroll]] for (uint kk = 0u; kk < TILE_K; kk += TK) {
            [[unroll]] for (uint cm_row = 0u; cm_row < CMS_PER_ROW; ++cm_row) {
                coopMatLoad(gate_w, shared_gate_w, int(cm_row * TM * SHMEM_STRIDE + kk / 2u), int(SHMEM_STRIDE), gl_CooperativeMatrixLayoutRowMajor);
                coopMatLoad(up_w, shared_up_w, int(cm_row * TM * SHMEM_STRIDE + kk / 2u), int(SHMEM_STRIDE), gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint cm_col = 0u; cm_col < CMS_PER_COL; ++cm_col) {
                    coopMatLoad(mat_x, shared_x, int((warp * WN + cm_col * TN) * SHMEM_STRIDE + kk / 2u), int(SHMEM_STRIDE), gl_CooperativeMatrixLayoutColumnMajor);
                    const uint sum_index = cm_col * CMS_PER_ROW + cm_row;
                    gate_sums[sum_index] = coopMatMulAdd(gate_w, mat_x, gate_sums[sum_index]);
                    up_sums[sum_index] = coopMatMulAdd(up_w, mat_x, up_sums[sum_index]);
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
            coopMatStore(gate_sums[sum_index], shared_gate_stage, int(warp * STAGE_SIZE), int(TM), gl_CooperativeMatrixLayoutColumnMajor);
            coopMatStore(up_sums[sum_index], shared_up_stage, int(warp * STAGE_SIZE), int(TM), gl_CooperativeMatrixLayoutColumnMajor);
            controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);

            [[unroll]] for (uint col = 0u; col < TN; col += store_stride) {
                const uint n = n_tile + store_r;
                const uint m = m_tile + col + store_c;
                if (n < pc.N && m < pc.M) {
                    const uint stage_index = warp * STAGE_SIZE + (col + store_c) * TM + store_r;
                    const float gate = float(shared_gate_stage[stage_index]);
                    const float up = float(shared_up_stage[stage_index]);
                    output_values[m * pc.N + n] = float16_t(swiglu_value(gate, up));
                }
            }
            controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
        }
    }
}
""",
)
