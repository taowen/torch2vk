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
)
from torch2vk.export.shaders.linear_bias_f32 import make_linear_bias_variant
from torch2vk.runtime.shader import (
    ExprDim,
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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements, SubgroupRequirements
from torch2vk.vulkan.types import q8_0_halfwords_layout


def make_linear_bias_q8_0_variant(node: Node) -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    w_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not x_shape or not w_shape or not out_shape:
        return None
    k = int(x_shape[-1])
    if k % 32 != 0:
        return make_linear_bias_variant(node)

    x_contract = tuple(f"X{i}" for i in range(len(x_shape) - 1)) + ("K",)
    b_contract = ("N",)
    out_contract = tuple(f"X{i}" for i in range(len(out_shape) - 1)) + ("N",)
    bias_dtype = node_input_dtype(node, 2)
    bias_suffix = weight_dtype_suffix(bias_dtype)
    concrete_m = 1
    for dim in x_shape[:-1]:
        concrete_m *= int(dim)
    if concrete_m <= 8:
        return ShaderVariant(
            name=f"linear_bias_q8_0w_{bias_suffix}b_matvec_f32",
            family="export",
            contract=_contract(
                x_contract=x_contract,
                b_contract=b_contract,
                out_contract=out_contract,
                bias_dtype=bias_dtype,
                shader_name=f"linear_bias_q8_0w_{bias_suffix}b_matvec_f32",
                class_name=f"ExportLinearBiasQ8_0Weight{bias_suffix.title()}BiasMatvecProgram",
                matvec=True,
            ),
            execution_requirements=_SUBGROUP64_16BIT_REQUIREMENTS,
            source=_matvec_source(bias_dtype),
        )
    return ShaderVariant(
        name=f"linear_bias_q8_0w_{bias_suffix}b_f32",
        family="export",
        contract=_contract(
            x_contract=x_contract,
            b_contract=b_contract,
            out_contract=out_contract,
            bias_dtype=bias_dtype,
            shader_name=f"linear_bias_q8_0w_{bias_suffix}b_f32",
            class_name=f"ExportLinearBiasQ8_0Weight{bias_suffix.title()}BiasProgram",
            matvec=False,
        ),
        execution_requirements=_COOPMAT_REQUIREMENTS,
        source=_coopmat_source(bias_dtype),
    )


def _contract(
    *,
    x_contract: tuple[ExprDim, ...],
    b_contract: tuple[ExprDim, ...],
    out_contract: tuple[ExprDim, ...],
    bias_dtype: str,
    shader_name: str,
    class_name: str,
    matvec: bool,
) -> ShaderContract:
    m = product_expr(tuple(x_contract[:-1]))
    return ShaderContract(
        class_name=class_name,
        shader_name=shader_name,
        fields=(
            TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=x_contract)),
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
            TensorFieldSpec("bias", IOKind.INPUT, "input", TensorContract(dtype=bias_dtype, shape=b_contract)),
            TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("M", PushConstantType.UINT32, 0, m),
                PushConstantFieldSpec("K", PushConstantType.UINT32, 4, "K"),
                PushConstantFieldSpec("N", PushConstantType.UINT32, 8, "N"),
            ),
        ),
        dispatch=(
            (ceil_div("N", 2), m, 1)
            if matvec
            else (ceil_div(m, 16), ceil_div("N", 16), 1)
        ),
    )


def _matvec_source(bias_dtype: str) -> str:
    return (
        _Q8_0_MATVEC_BIAS_SOURCE
        .replace("{{BIAS_EXTENSION}}", weight_extension_source("bfloat16") if bias_dtype == "bfloat16" else "")
        .replace("{{BIAS_TYPE}}", weight_glsl_type(bias_dtype))
    )


def _coopmat_source(bias_dtype: str) -> str:
    return (
        _Q8_0_COOPMAT_BIAS_SOURCE
        .replace("{{BIAS_EXTENSION}}", weight_extension_source("bfloat16") if bias_dtype == "bfloat16" else "")
        .replace("{{BIAS_TYPE}}", weight_glsl_type(bias_dtype))
    )


_SUBGROUP64_16BIT_REQUIREMENTS = ShaderExecutionRequirements(
    subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
    require_storage_buffer_16bit_access=True,
)
_COOPMAT_REQUIREMENTS = ShaderExecutionRequirements(
    subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
    require_storage_buffer_16bit_access=True,
)


_Q8_0_MATVEC_BIAS_SOURCE = """\
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
{{BIAS_EXTENSION}}\

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { {{BIAS_TYPE}} bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

float q8_0_value(uint row, uint k) {
    const uint blocks_per_row = pc.K / 32u;
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

void main() {
    const uint lane = gl_SubgroupInvocationID;
    const uint row = gl_WorkGroupID.y;
    const uint col0 = gl_WorkGroupID.x * 2u;
    const uint col1 = col0 + 1u;
    float acc0 = 0.0;
    float acc1 = 0.0;
    if (row < pc.M) {
        for (uint k = lane; k < pc.K; k += 64u) {
            const float x_value = x[row * pc.K + k];
            if (col0 < pc.N) { acc0 = fma(x_value, q8_0_value(col0, k), acc0); }
            if (col1 < pc.N) { acc1 = fma(x_value, q8_0_value(col1, k), acc1); }
        }
    }
    acc0 = subgroupAdd(acc0);
    acc1 = subgroupAdd(acc1);
    if (lane == 0u && row < pc.M) {
        if (col0 < pc.N) { output_values[row * pc.N + col0] = acc0 + float(bias[col0]); }
        if (col1 < pc.N) { output_values[row * pc.N + col1] = acc1 + float(bias[col1]); }
    }
}
"""


_Q8_0_COOPMAT_BIAS_SOURCE = """\
#version 460

#pragma use_vulkan_memory_model

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_cooperative_matrix : require
{{BIAS_EXTENSION}}\

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { {{BIAS_TYPE}} bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const uint TILE_M = 16u;
const uint TILE_N = 16u;
const uint TILE_K = 16u;
const uint TILE_SIZE = TILE_M * TILE_K;

shared float16_t shared_a[TILE_SIZE];
shared float16_t shared_b[TILE_SIZE];
shared float shared_out[TILE_M * TILE_N];

float q8_0_value(uint row, uint k) {
    const uint blocks_per_row = pc.K / 32u;
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

void load_a_tile(uint lane, uint row_base, uint k_base) {
    for (uint i = lane; i < TILE_SIZE; i += 64u) {
        const uint row = i / TILE_K;
        const uint col = i - row * TILE_K;
        const uint m = row_base + row;
        const uint k = k_base + col;
        shared_a[i] = float16_t((m < pc.M && k < pc.K) ? x[m * pc.K + k] : 0.0);
    }
}

void load_b_tile(uint lane, uint col_base, uint k_base) {
    for (uint i = lane; i < TILE_SIZE; i += 64u) {
        const uint col = i / TILE_K;
        const uint k_offset = i - col * TILE_K;
        const uint n = col_base + col;
        const uint k = k_base + k_offset;
        shared_b[i] = float16_t((n < pc.N && k < pc.K) ? q8_0_value(n, k) : 0.0);
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

    for (uint k_base = 0u; k_base < pc.K; k_base += TILE_K) {
        load_a_tile(lane, row_base, k_base);
        load_b_tile(lane, col_base, k_base);
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
            output_values[m * pc.N + n] = shared_out[i] + float(bias[n]);
        }
    }
}
"""
