"""Generated shader: conv_transpose1d_q8_0w_f32b_f32_40."""

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


CONV_TRANSPOSE1D_Q8_0W_F32B_F32_40 = ShaderVariant(
    name='conv_transpose1d_q8_0w_f32b_f32_40',
    family='export',
    contract=ShaderContract(
        class_name='ExportConvTranspose1dQ8_0WeightF32BiasProgram',
        shader_name='conv_transpose1d_q8_0w_f32b_f32_40',
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
                contract=TensorContract(dtype='uint16', shape=('Ci', 136,), layout=q8_0_halfwords_layout(logical_k=256, block_size=32, halfwords_per_block=17)),
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
                PushConstantFieldSpec('Kh', PushConstantType.UINT32, 20, 4, dynamic=False),
                PushConstantFieldSpec('stride', PushConstantType.UINT32, 24, 2, dynamic=False),
                PushConstantFieldSpec('padding', PushConstantType.UINT32, 28, 1, dynamic=False),
                PushConstantFieldSpec('dilation', PushConstantType.UINT32, 32, 1, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(ceil_div('Lo', 2), 32), ceil_div('Co', 16), mul('B', 2)),
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

uint taps_per_output() {
    return (pc.Kh + pc.stride - 1u) / pc.stride;
}

uint logical_k() {
    return pc.Ci * taps_per_output();
}

float q8_0_value(uint row, uint k) {
    const uint kernel_k = pc.Co * pc.Kh;
    const uint blocks_per_row = (kernel_k + 31u) / 32u;
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

int first_kernel_for_residue(uint residue) {
    int first_k = (int(residue) + int(pc.padding)) % int(pc.stride);
    if (first_k < 0) {
        first_k += int(pc.stride);
    }
    return first_k;
}

float deconv_input_value(uint b, uint residue, uint u, uint k_eff) {
    const uint taps = taps_per_output();
    const uint ic = k_eff / taps;
    const uint tap = k_eff - ic * taps;
    const int kernel = first_kernel_for_residue(residue) + int(tap * pc.stride);
    if (b >= pc.B || ic >= pc.Ci || kernel < 0 || kernel >= int(pc.Kh)) {
        return 0.0;
    }
    const uint t = residue + u * pc.stride;
    if (t >= pc.Lo) {
        return 0.0;
    }
    const int it_signed = (int(t) + int(pc.padding) - kernel) / int(pc.stride);
    if (it_signed < 0 || it_signed >= int(pc.Li)) {
        return 0.0;
    }
    return float(x[(b * pc.Ci + ic) * pc.Li + uint(it_signed)]);
}

float deconv_weight_value(uint residue, uint n, uint k_eff) {
    const uint taps = taps_per_output();
    const uint ic = k_eff / taps;
    const uint tap = k_eff - ic * taps;
    const int kernel = first_kernel_for_residue(residue) + int(tap * pc.stride);
    if (n >= pc.Co || ic >= pc.Ci || kernel < 0 || kernel >= int(pc.Kh)) {
        return 0.0;
    }
    return q8_0_value(ic, n * pc.Kh + uint(kernel));
}

void load_a_tile_pair(uint lane, uint b, uint residue, uint row_base, uint k_base) {
    const uint kernel_k = logical_k();
    for (uint i = lane; i < TILE_SIZE; i += 64u) {
        const uint row = i / TILE_K;
        const uint col = i - row * TILE_K;
        const uint u0 = row_base + row;
        const uint u1 = row_base + TILE_M + row;
        const uint k = k_base + col;
        shared_a0[i] = float16_t(k < kernel_k ? deconv_input_value(b, residue, u0, k) : 0.0);
        shared_a1[i] = float16_t(k < kernel_k ? deconv_input_value(b, residue, u1, k) : 0.0);
    }
}

void load_b_tile(uint lane, uint residue, uint col_base, uint k_base) {
    const uint kernel_k = logical_k();
    for (uint i = lane; i < TILE_SIZE; i += 64u) {
        const uint col = i / TILE_K;
        const uint k_offset = i - col * TILE_K;
        const uint n = col_base + col;
        const uint k = k_base + k_offset;
        shared_b[i] = float16_t(k < kernel_k ? deconv_weight_value(residue, n, k) : 0.0);
    }
}

void main() {
    const uint lane = gl_SubgroupInvocationID;
    const uint batch_residue = gl_WorkGroupID.z;
    const uint b = batch_residue / pc.stride;
    const uint residue = batch_residue - b * pc.stride;
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
        load_a_tile_pair(lane, b, residue, row_base, k_base);
        load_b_tile(lane, residue, col_base, k_base);
        barrier();
        coopMatLoad(mat_a0, shared_a0, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(mat_a1, shared_a1, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(mat_b, shared_b, 0, int(TILE_K), gl_CooperativeMatrixLayoutColumnMajor);
        mat_c0 = coopMatMulAdd(mat_a0, mat_b, mat_c0);
        mat_c1 = coopMatMulAdd(mat_a1, mat_b, mat_c1);
        barrier();

        load_a_tile_pair(lane, b, residue, row_base, k_base + TILE_K);
        load_b_tile(lane, residue, col_base, k_base + TILE_K);
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
        const uint u0 = row_base + row;
        const uint u1 = row_base + TILE_M + row;
        const uint n = col_base + col;
        if (b < pc.B && n < pc.Co) {
            const uint t0 = residue + u0 * pc.stride;
            if (t0 < pc.Lo) {
                float acc = shared_out0[i] + float(bias[n]);
                output_values[(b * pc.Co + n) * pc.Lo + t0] = float16_t(acc);
            }
            const uint t1 = residue + u1 * pc.stride;
            if (t1 < pc.Lo) {
                float acc = shared_out1[i] + float(bias[n]);
                output_values[(b * pc.Co + n) * pc.Lo + t1] = float16_t(acc);
            }
        }
    }
}
""",
)
