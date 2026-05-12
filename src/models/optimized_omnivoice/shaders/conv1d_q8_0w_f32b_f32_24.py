"""Generated shader: conv1d_q8_0w_f32b_f32_24."""

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
    q8_0_halfwords_layout,
)


CONV1D_Q8_0W_F32B_F32_24 = ShaderVariant(
    name='conv1d_q8_0w_f32b_f32_24',
    family='export',
    contract=ShaderContract(
        class_name='ExportConv1dQ8_0WeightF32BiasProgram',
        shader_name='conv1d_q8_0w_f32b_f32_24',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('B', 'Ci', 'Li',)),
            ),
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='uint16', shape=('Co', 272,), layout=q8_0_halfwords_layout(logical_k=512, block_size=32, halfwords_per_block=17)),
            ),
            TensorFieldSpec(
                name='bias',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('Co',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('B', 'Co', 'Lo',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=36,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('Ci', PushConstantType.UINT32, 4, 'Ci', dynamic=False),
                PushConstantFieldSpec('Li', PushConstantType.UINT32, 8, 'Li', dynamic=False),
                PushConstantFieldSpec('Co', PushConstantType.UINT32, 12, 'Co', dynamic=False),
                PushConstantFieldSpec('Lo', PushConstantType.UINT32, 16, 'Lo', dynamic=False),
                PushConstantFieldSpec('Kh', PushConstantType.UINT32, 20, 1, dynamic=False),
                PushConstantFieldSpec('stride', PushConstantType.UINT32, 24, 1, dynamic=False),
                PushConstantFieldSpec('padding', PushConstantType.UINT32, 28, 0, dynamic=False),
                PushConstantFieldSpec('dilation', PushConstantType.UINT32, 32, 1, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul('B', 'Lo'), 32), ceil_div('Co', 16), 1),
    ),
    execution_requirements=ShaderExecutionRequirements(subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True), cooperative_matrix=CooperativeMatrixRequirements(scope='subgroup', m_size=16, n_size=16, k_size=16, a_type='float16', b_type='float16', c_type='float32', result_type='float32', saturating_accumulation=False), require_storage_buffer_16bit_access=True),
    source="""\
#version 460

#pragma use_vulkan_memory_model

#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { float bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants {
    uint B; uint Ci; uint Li; uint Co; uint Lo; uint Kh;
    uint stride; uint padding; uint dilation;
} pc;
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
shared float shared_q8_d[TILE_N];

uint logical_k() {
    return pc.Ci * pc.Kh;
}

void prepare_q8_0_tile_scales(uint lane, uint col_base, uint k_base) {
    const uint kernel_k = logical_k();
    const uint blocks_per_row = (kernel_k + 31u) / 32u;
    const uint block_index = k_base >> 5u;
    for (uint col = lane; col < TILE_N; col += 64u) {
        const uint n = col_base + col;
        if (n < pc.Co) {
            const uint block_half = n * blocks_per_row * 17u + block_index * 17u;
            shared_q8_d[col] = unpackHalf2x16(uint(weight[block_half])).x;
        } else {
            shared_q8_d[col] = 0.0;
        }
    }
}

float q8_0_value_prepared(uint row, uint k) {
    const uint kernel_k = logical_k();
    const uint blocks_per_row = (kernel_k + 31u) / 32u;
    const uint block_index = k >> 5u;
    const uint block_half = row * blocks_per_row * 17u + block_index * 17u;
    const uint local = k & 31u;
    const uint packed = uint(weight[block_half + 1u + (local >> 1u)]);
    uint byte_value = ((local & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
    int quant = int(byte_value);
    if (quant >= 128) { quant -= 256; }
    return shared_q8_d[row & 15u] * float(quant);
}

float conv_input_value(uint m, uint k) {
    const uint b = m / pc.Lo;
    const uint t = m - b * pc.Lo;
    const uint ic = k / pc.Kh;
    const uint kh = k - ic * pc.Kh;
    const uint padded_pos = t * pc.stride + kh * pc.dilation;
    if (b >= pc.B || ic >= pc.Ci || padded_pos < pc.padding) {
        return 0.0;
    }
    const uint it = padded_pos - pc.padding;
    if (it >= pc.Li) {
        return 0.0;
    }
    return float(x[(b * pc.Ci + ic) * pc.Li + it]);
}

void load_a_tile_pair(uint lane, uint row_base, uint k_base) {
    const uint kernel_k = logical_k();
    for (uint i = lane; i < TILE_SIZE; i += 64u) {
        const uint row = i / TILE_K;
        const uint col = i - row * TILE_K;
        const uint m0 = row_base + row;
        const uint m1 = row_base + TILE_M + row;
        const uint k = k_base + col;
        shared_a0[i] = float16_t((m0 < pc.B * pc.Lo && k < kernel_k) ? conv_input_value(m0, k) : 0.0);
        shared_a1[i] = float16_t((m1 < pc.B * pc.Lo && k < kernel_k) ? conv_input_value(m1, k) : 0.0);
    }
}

void load_b_tile(uint lane, uint col_base, uint k_base) {
    const uint kernel_k = logical_k();
    for (uint i = lane; i < TILE_SIZE; i += 64u) {
        const uint col = i / TILE_K;
        const uint k_offset = i - col * TILE_K;
        const uint n = col_base + col;
        const uint k = k_base + k_offset;
        shared_b[i] = float16_t((n < pc.Co && k < kernel_k) ? q8_0_value_prepared(n, k) : 0.0);
    }
}

void main() {
    const uint lane = gl_SubgroupInvocationID;
    const uint row_base = gl_WorkGroupID.x * GROUP_M;
    const uint col_base = gl_WorkGroupID.y * TILE_N;
    const uint kernel_k = logical_k();

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_a0;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_a1;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_b;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> mat_c0;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> mat_c1;
    mat_c0 = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    mat_c1 = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint k_base = 0u; k_base < kernel_k; k_base += 32u) {
        prepare_q8_0_tile_scales(lane, col_base, k_base);
        barrier();

        load_a_tile_pair(lane, row_base, k_base);
        load_b_tile(lane, col_base, k_base);
        barrier();
        coopMatLoad(mat_a0, shared_a0, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(mat_a1, shared_a1, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(mat_b, shared_b, 0, int(TILE_K), gl_CooperativeMatrixLayoutColumnMajor);
        mat_c0 = coopMatMulAdd(mat_a0, mat_b, mat_c0);
        mat_c1 = coopMatMulAdd(mat_a1, mat_b, mat_c1);
        barrier();

        load_a_tile_pair(lane, row_base, k_base + TILE_K);
        load_b_tile(lane, col_base, k_base + TILE_K);
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
        if (n < pc.Co) {
            const uint b0 = m0 / pc.Lo;
            const uint t0 = m0 - b0 * pc.Lo;
            const uint b1 = m1 / pc.Lo;
            const uint t1 = m1 - b1 * pc.Lo;
            if (m0 < pc.B * pc.Lo) {
                float acc = shared_out0[i] + float(bias[n]);
                output_values[(b0 * pc.Co + n) * pc.Lo + t0] = float16_t(acc);
            }
            if (m1 < pc.B * pc.Lo) {
                float acc = shared_out1[i] + float(bias[n]);
                output_values[(b1 * pc.Co + n) * pc.Lo + t1] = float16_t(acc);
            }
        }
    }
}
""",
)
