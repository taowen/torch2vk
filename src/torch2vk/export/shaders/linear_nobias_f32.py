from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    node_input_dtype,
    node_input_shape,
    node_output_shape,
    product_expr,
    weight_dtype_suffix,
    weight_extension_source,
    weight_glsl_type,
    weight_zero_literal,
)
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

_SOURCE_TEMPLATE = """\
#version 450
#extension GL_EXT_control_flow_attributes : enable
{{WEIGHT_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { {{WEIGHT_TYPE}} weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
const uint TILE_M = 16u; const uint TILE_N = 64u; const uint TILE_K = 32u;
shared float tile_x[16 * 32]; shared {{WEIGHT_TYPE}} tile_w[32 * 64];
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
            const uint tr = i / TILE_K; const uint tk = i - tr * TILE_K;
            const uint gr = gl_WorkGroupID.x * TILE_M + tr; const uint gk = k0 + tk;
            tile_x[i] = (gr < pc.M && gk < pc.K) ? x[gr * pc.K + gk] : 0.0;
        }
        for (uint i = lane; i < TILE_K * TILE_N; i += 256u) {
            const uint tk = i / TILE_N; const uint tc = i - tk * TILE_N;
            const uint gk = k0 + tk; const uint gc = gl_WorkGroupID.y * TILE_N + tc;
            tile_w[i] = (gc < pc.N && gk < pc.K) ? weight[gc * pc.K + gk] : {{WEIGHT_ZERO}};
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
"""


def make_linear_nobias_variant(node: Node) -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    w_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not x_shape or not w_shape or not out_shape:
        return None

    x_contract = tuple(f"X{i}" for i in range(len(x_shape)))
    w_contract = tuple(f"W{i}" for i in range(len(w_shape)))
    out_contract = tuple(f"Y{i}" for i in range(len(out_shape)))

    weight_dtype = node_input_dtype(node, 1)
    weight_suffix = weight_dtype_suffix(weight_dtype)
    shader_name = f"linear_nobias_{weight_suffix}w_f32"
    m = product_expr(tuple(x_contract[:-1]))

    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"ExportLinearNobias{weight_suffix.title()}WeightProgram",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=x_contract)),
                TensorFieldSpec("weight", IOKind.INPUT, "weight", TensorContract(dtype=weight_dtype, shape=w_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=12,
                fields=(
                    PushConstantFieldSpec("M", PushConstantType.UINT32, 0, m),
                    PushConstantFieldSpec("K", PushConstantType.UINT32, 4, x_contract[-1]),
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 8, w_contract[0]),
                ),
            ),
            dispatch=(ceil_div(m, 16), ceil_div(w_contract[0], 64), 1),
        ),
        source=_source(weight_dtype),
    )


def _source(weight_dtype: str) -> str:
    return (
        _SOURCE_TEMPLATE
        .replace("{{WEIGHT_EXTENSION}}", weight_extension_source(weight_dtype))
        .replace("{{WEIGHT_TYPE}}", weight_glsl_type(weight_dtype))
        .replace("{{WEIGHT_ZERO}}", weight_zero_literal(weight_dtype))
    )
