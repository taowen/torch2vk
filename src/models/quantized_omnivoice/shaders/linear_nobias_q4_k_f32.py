"""Q4_K weight linear shader for quantized OmniVoice."""

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
from torch2vk.vulkan.types import q4_k_words_layout


LINEAR_NOBIAS_Q4_K_F32 = ShaderVariant(
    name="linear_nobias_q4_k_f32",
    family="quantized_omnivoice",
    contract=ShaderContract(
        class_name="LinearNobiasQ4KProgram",
        shader_name="linear_nobias_q4_k_f32",
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
                    dtype="uint32",
                    shape=("N", mul(ceil_div("K", 256), 36)),
                    layout=q4_k_words_layout(logical_k="K"),
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
    source="""\
#version 450

#extension GL_EXT_control_flow_attributes : enable

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint weight[]; };
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
            tile_w[i] = (gc < pc.N && gk < pc.K) ? q4k_value(gc, gk) : 0.0;
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
