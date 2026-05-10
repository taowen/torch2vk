"""Q8_0 weight linear shader for quantized OmniVoice."""

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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements
from torch2vk.vulkan.types import q8_0_halfwords_layout


LINEAR_NOBIAS_Q8_0_F32 = ShaderVariant(
    name="linear_nobias_q8_0_f32",
    family="quantized_omnivoice",
    contract=ShaderContract(
        class_name="LinearNobiasQ8_0Program",
        shader_name="linear_nobias_q8_0_f32",
        fields=(
            TensorFieldSpec(
                "x",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float32", shape=("X0", "X1", "K")),
            ),
            TensorFieldSpec(
                "weight",
                IOKind.INPUT,
                "weight",
                TensorContract(
                    dtype="uint16",
                    shape=("N", mul(ceil_div("K", 32), 17)),
                    layout=q8_0_halfwords_layout(logical_k="K"),
                ),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float32", shape=("X0", "X1", "N")),
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
        dispatch=(ceil_div(mul("X0", "X1"), 16), ceil_div("N", 64), 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants {
    uint M;
    uint K;
    uint N;
} pc;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

const uint TILE_M = 16u;
const uint TILE_N = 64u;
const uint TILE_K = 32u;

shared float tile_x[16 * 32];
shared float tile_w[32 * 64];

float q8_0_value(uint row, uint k) {
    const uint blocks_per_row = pc.K / 32u;
    const uint block_index = k >> 5u;
    const uint block_half = row * blocks_per_row * 17u + block_index * 17u;
    const float d = unpackHalf2x16(uint(weight[block_half])).x;
    const uint local = k & 31u;
    const uint packed = uint(weight[block_half + 1u + (local >> 1u)]);
    uint byte_value = ((local & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
    int quant = int(byte_value);
    if (quant >= 128) {
        quant -= 256;
    }
    return d * float(quant);
}

void main() {
    const uint local_col = gl_LocalInvocationID.x;
    const uint local_row = gl_LocalInvocationID.y;
    const uint lane = local_row * 16u + local_col;
    const uint row = gl_WorkGroupID.x * TILE_M + local_row;
    const uint col0 = gl_WorkGroupID.y * TILE_N + local_col;
    const uint col1 = col0 + 16u;
    const uint col2 = col0 + 32u;
    const uint col3 = col0 + 48u;

    float acc0 = 0.0;
    float acc1 = 0.0;
    float acc2 = 0.0;
    float acc3 = 0.0;

    for (uint k0 = 0u; k0 < pc.K; k0 += TILE_K) {
        for (uint i = lane; i < TILE_M * TILE_K; i += 256u) {
            const uint tr = i / TILE_K;
            const uint tk = i - tr * TILE_K;
            const uint gr = gl_WorkGroupID.x * TILE_M + tr;
            const uint gk = k0 + tk;
            tile_x[i] = (gr < pc.M && gk < pc.K) ? x[gr * pc.K + gk] : 0.0;
        }
        for (uint i = lane; i < TILE_K * TILE_N; i += 256u) {
            const uint tk = i / TILE_N;
            const uint tc = i - tk * TILE_N;
            const uint gk = k0 + tk;
            const uint gc = gl_WorkGroupID.y * TILE_N + tc;
            tile_w[i] = (gc < pc.N && gk < pc.K) ? q8_0_value(gc, gk) : 0.0;
        }
        barrier();

        [[unroll]] for (uint k = 0u; k < TILE_K; ++k) {
            const float x_value = tile_x[local_row * TILE_K + k];
            acc0 = fma(x_value, tile_w[k * TILE_N + local_col], acc0);
            acc1 = fma(x_value, tile_w[k * TILE_N + local_col + 16u], acc1);
            acc2 = fma(x_value, tile_w[k * TILE_N + local_col + 32u], acc2);
            acc3 = fma(x_value, tile_w[k * TILE_N + local_col + 48u], acc3);
        }
        barrier();
    }

    if (row < pc.M && col0 < pc.N) { output_values[row * pc.N + col0] = acc0; }
    if (row < pc.M && col1 < pc.N) { output_values[row * pc.N + col1] = acc1; }
    if (row < pc.M && col2 < pc.N) { output_values[row * pc.N + col2] = acc2; }
    if (row < pc.M && col3 < pc.N) { output_values[row * pc.N + col3] = acc3; }
}
""",
)
