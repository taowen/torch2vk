from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    activation_store,
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
{{ACTIVATION_EXTENSION}}\
{{WEIGHT_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { {{WEIGHT_TYPE}} weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { {{BIAS_TYPE}} bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
const uint TILE_M = 16u; const uint TILE_N = 16u; const uint TILE_K = 32u;
shared {{ACTIVATION_TYPE}} tile_x[16 * 32]; shared {{WEIGHT_TYPE}} tile_w[32 * 16];
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
            tile_x[i] = {{TILE_X_STORE}};
        }
        for (uint i = lane; i < TILE_K * TILE_N; i += TILE_M * TILE_N) {
            const uint tk = i / TILE_N; const uint tc = i - tk * TILE_N;
            const uint gk = k0 + tk; const uint gc = gl_WorkGroupID.y * TILE_N + tc;
            tile_w[i] = (gc < pc.N && gk < pc.K) ? weight[gc * pc.K + gk] : {{WEIGHT_ZERO}};
        }
        barrier();
        [[unroll]] for (uint k = 0u; k < TILE_K; ++k) {
            acc = fma(float(tile_x[local_row * TILE_K + k]), float(tile_w[k * TILE_N + local_col]), acc);
        }
        barrier();
    }
    if (row < pc.M && col < pc.N) { output_values[row * pc.N + col] = {{STORE_ACC}}; }
}
"""


def make_linear_bias_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    w_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not x_shape or not w_shape or not out_shape:
        return None

    x_contract = tuple(f"X{i}" for i in range(len(x_shape)))
    w_contract = tuple(f"W{i}" for i in range(len(w_shape)))
    b_contract = ("N",)
    out_contract = tuple(f"Y{i}" for i in range(len(out_shape)))

    weight_dtype = node_input_dtype(node, 1)
    bias_dtype = node_input_dtype(node, 2)
    weight_suffix = weight_dtype_suffix(weight_dtype)
    bias_suffix = weight_dtype_suffix(bias_dtype)
    shader_name = f"linear_bias_{weight_suffix}w_{bias_suffix}b_f32"
    m = product_expr(tuple(x_contract[:-1]))

    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"ExportLinearBias{weight_suffix.title()}Weight{bias_suffix.title()}BiasProgram",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=x_contract),
                ),
                TensorFieldSpec(
                    "weight",
                    IOKind.INPUT,
                    "weight",
                    TensorContract(dtype=weight_dtype, shape=w_contract),
                ),
                TensorFieldSpec(
                    "bias",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=bias_dtype, shape=b_contract),
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=out_contract),
                ),
            ),
            push_constants=PushConstantSpec(
                size=12,
                fields=(
                    PushConstantFieldSpec("M", PushConstantType.UINT32, 0, m),
                    PushConstantFieldSpec("K", PushConstantType.UINT32, 4, x_contract[-1]),
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 8, w_contract[0]),
                ),
            ),
            dispatch=(ceil_div(m, 16), ceil_div(w_contract[0], 16), 1),
        ),
        source=_source(
            weight_dtype=weight_dtype, bias_dtype=bias_dtype, activation_dtype=activation_dtype
        ),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _source(*, weight_dtype: str, bias_dtype: str, activation_dtype: str) -> str:
    extension = (
        weight_extension_source("bfloat16") if "bfloat16" in {weight_dtype, bias_dtype} else ""
    )
    return (
        _SOURCE_TEMPLATE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{WEIGHT_EXTENSION}}", extension)
        .replace("{{WEIGHT_TYPE}}", weight_glsl_type(weight_dtype))
        .replace("{{BIAS_TYPE}}", weight_glsl_type(bias_dtype))
        .replace("{{WEIGHT_ZERO}}", weight_zero_literal(weight_dtype))
        .replace(
            "{{TILE_X_STORE}}",
            activation_store(
                "(gr < pc.M && gk < pc.K) ? x[gr * pc.K + gk] : 0.0", activation_dtype
            ),
        )
        .replace("{{STORE_ACC}}", activation_store("acc + float(bias[col])", activation_dtype))
    )
