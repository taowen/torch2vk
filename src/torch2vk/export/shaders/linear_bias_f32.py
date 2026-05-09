from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import node_input_shape, node_output_shape
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

_SOURCE = """\
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
"""


def make_linear_bias_variant(node: Node) -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    w_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not x_shape or not w_shape or not out_shape:
        return None

    x_contract = tuple(f"X{i}" for i in range(len(x_shape)))
    w_contract = tuple(f"W{i}" for i in range(len(w_shape)))
    b_contract = ("N",)
    out_contract = tuple(f"Y{i}" for i in range(len(out_shape)))

    k = x_shape[-1]
    n = w_shape[0]
    m = 1
    for d in x_shape[:-1]:
        m *= d

    return ShaderVariant(
        name="linear_bias_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportLinearBiasProgram",
            shader_name="linear_bias_f32",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=x_contract)),
                TensorFieldSpec("weight", IOKind.INPUT, "weight", TensorContract(dtype="bfloat16", shape=w_contract)),
                TensorFieldSpec("bias", IOKind.INPUT, "input", TensorContract(dtype="bfloat16", shape=b_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=12,
                fields=(
                    PushConstantFieldSpec("M", PushConstantType.UINT32, 0, m),
                    PushConstantFieldSpec("K", PushConstantType.UINT32, 4, k),
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 8, n),
                ),
            ),
            dispatch=(ceil_div(m, 16), ceil_div(n, 16), 1),
        ),
        source=_SOURCE,
    )
