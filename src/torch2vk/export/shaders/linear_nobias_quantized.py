from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import node_input_shape, node_output_shape
from torch2vk.export.shaders.linear_nobias_f32 import make_linear_nobias_variant
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
from torch2vk.vulkan.types import q4_k_words_layout, q8_0_halfwords_layout


def make_linear_nobias_q4_k_m_variant(node: Node) -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    w_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not x_shape or not w_shape or not out_shape:
        return None
    k = int(x_shape[-1])
    m = _flattened_rows(x_shape)
    if k % 256 == 0:
        if m <= 8:
            return _make_quantized_variant(
                name="linear_nobias_q4_k_matvec_f32",
                class_name="ExportLinearNobiasQ4KMatvecProgram",
                x_shape=x_shape,
                w_shape=w_shape,
                out_shape=out_shape,
                weight_contract=TensorContract(
                    dtype="uint32",
                    shape=("N", mul(ceil_div("K", 256), 36)),
                    layout=q4_k_words_layout(logical_k="K"),
                ),
                source=_Q4_K_MATVEC_SOURCE,
                dispatch=(ceil_div(int(w_shape[0]), 2), m, 1),
                execution_requirements=_SUBGROUP64_REQUIREMENTS,
            )
        return _make_quantized_variant(
            name="linear_nobias_q4_k_f32",
            class_name="ExportLinearNobiasQ4KProgram",
            x_shape=x_shape,
            w_shape=w_shape,
            out_shape=out_shape,
            weight_contract=TensorContract(
                dtype="uint32",
                shape=("N", mul(ceil_div("K", 256), 36)),
                layout=q4_k_words_layout(logical_k="K"),
            ),
            source=_Q4_K_COOPMAT_SOURCE,
            dispatch=(ceil_div(m, 16), ceil_div(int(w_shape[0]), 16), 1),
            execution_requirements=_COOPMAT_REQUIREMENTS,
        )
    if k % 32 == 0:
        if m <= 8:
            return _make_q8_0_matvec_variant(x_shape=x_shape, w_shape=w_shape, out_shape=out_shape)
        return _make_quantized_variant(
            name="linear_nobias_q8_0_f32",
            class_name="ExportLinearNobiasQ8_0Program",
            x_shape=x_shape,
            w_shape=w_shape,
            out_shape=out_shape,
            weight_contract=TensorContract(
                dtype="uint16",
                shape=("N", mul(ceil_div("K", 32), 17)),
                layout=q8_0_halfwords_layout(logical_k="K"),
            ),
            source=_Q8_0_COOPMAT_SOURCE,
            dispatch=(ceil_div(m, 16), ceil_div(int(w_shape[0]), 16), 1),
            execution_requirements=_COOPMAT_REQUIREMENTS,
        )
    return make_linear_nobias_variant(node)


def make_linear_nobias_q8_0_variant(node: Node) -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    w_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not x_shape or not w_shape or not out_shape:
        return None
    k = int(x_shape[-1])
    if k % 32 != 0:
        return make_linear_nobias_variant(node)
    m = _flattened_rows(x_shape)
    if m <= 8:
        return _make_q8_0_matvec_variant(x_shape=x_shape, w_shape=w_shape, out_shape=out_shape)
    return _make_quantized_variant(
        name="linear_nobias_q8_0_f32",
        class_name="ExportLinearNobiasQ8_0Program",
        x_shape=x_shape,
        w_shape=w_shape,
        out_shape=out_shape,
        weight_contract=TensorContract(
            dtype="uint16",
            shape=("N", mul(ceil_div("K", 32), 17)),
            layout=q8_0_halfwords_layout(logical_k="K"),
        ),
        source=_Q8_0_COOPMAT_SOURCE,
        dispatch=(ceil_div(m, 16), ceil_div(int(w_shape[0]), 16), 1),
        execution_requirements=_COOPMAT_REQUIREMENTS,
    )


def _make_quantized_variant(
    *,
    name: str,
    class_name: str,
    x_shape: tuple[int, ...],
    w_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    weight_contract: TensorContract,
    source: str,
    dispatch: tuple[ExprDim, ExprDim, ExprDim],
    execution_requirements: ShaderExecutionRequirements | None = None,
) -> ShaderVariant:
    x_contract = tuple(f"X{i}" for i in range(len(x_shape) - 1)) + ("K",)
    out_contract = tuple(f"X{i}" for i in range(len(out_shape) - 1)) + ("N",)
    m = 1
    for dim in x_shape[:-1]:
        m *= int(dim)
    return ShaderVariant(
        name=name,
        family="export",
        contract=ShaderContract(
            class_name=class_name,
            shader_name=name,
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=x_contract)),
                TensorFieldSpec("weight", IOKind.INPUT, "weight", weight_contract),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=12,
                fields=(
                    PushConstantFieldSpec("M", PushConstantType.UINT32, 0, m),
                    PushConstantFieldSpec("K", PushConstantType.UINT32, 4, int(x_shape[-1])),
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 8, int(w_shape[0])),
                ),
            ),
            dispatch=dispatch,
        ),
        execution_requirements=execution_requirements,
        source=source,
    )


def _make_q8_0_matvec_variant(
    *,
    x_shape: tuple[int, ...],
    w_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> ShaderVariant:
    return _make_quantized_variant(
        name="linear_nobias_q8_0_matvec_f32",
        class_name="ExportLinearNobiasQ8_0MatvecProgram",
        x_shape=x_shape,
        w_shape=w_shape,
        out_shape=out_shape,
        weight_contract=TensorContract(
            dtype="uint16",
            shape=("N", mul(ceil_div("K", 32), 17)),
            layout=q8_0_halfwords_layout(logical_k="K"),
        ),
        source=_Q8_0_MATVEC_SOURCE,
        dispatch=(ceil_div(int(w_shape[0]), 2), _flattened_rows(x_shape), 1),
        execution_requirements=_SUBGROUP64_16BIT_REQUIREMENTS,
    )


def _flattened_rows(x_shape: tuple[int, ...]) -> int:
    m = 1
    for dim in x_shape[:-1]:
        m *= int(dim)
    return m


_SUBGROUP64_REQUIREMENTS = ShaderExecutionRequirements(
    subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
)
_SUBGROUP64_16BIT_REQUIREMENTS = ShaderExecutionRequirements(
    subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
    require_storage_buffer_16bit_access=True,
)
_COOPMAT_REQUIREMENTS = ShaderExecutionRequirements(
    subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
    require_storage_buffer_16bit_access=True,
)


_Q4_K_MATVEC_SOURCE = """\
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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
    const uint lane = gl_SubgroupInvocationID;
    const uint row = gl_WorkGroupID.y;
    const uint col0 = gl_WorkGroupID.x * 2u;
    const uint col1 = col0 + 1u;
    float acc0 = 0.0;
    float acc1 = 0.0;
    if (row < pc.M) {
        for (uint k = lane; k < pc.K; k += 64u) {
            const float x_value = x[row * pc.K + k];
            if (col0 < pc.N) { acc0 = fma(x_value, q4k_value(col0, k), acc0); }
            if (col1 < pc.N) { acc1 = fma(x_value, q4k_value(col1, k), acc1); }
        }
    }
    acc0 = subgroupAdd(acc0);
    acc1 = subgroupAdd(acc1);
    if (lane == 0u && row < pc.M) {
        if (col0 < pc.N) { output_values[row * pc.N + col0] = acc0; }
        if (col1 < pc.N) { output_values[row * pc.N + col1] = acc1; }
    }
}
"""


_Q8_0_MATVEC_SOURCE = """\
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

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
        if (col0 < pc.N) { output_values[row * pc.N + col0] = acc0; }
        if (col1 < pc.N) { output_values[row * pc.N + col1] = acc1; }
    }
}
"""


_COOPMAT_COMMON_HEADER = """\
#version 460

#pragma use_vulkan_memory_model

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_cooperative_matrix : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
{{WEIGHT_BUFFER}}
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const uint TILE_M = 16u;
const uint TILE_N = 16u;
const uint TILE_K = 16u;
const uint TILE_SIZE = TILE_M * TILE_K;

shared float16_t shared_a[TILE_SIZE];
shared float16_t shared_b[TILE_SIZE];
shared float shared_out[TILE_M * TILE_N];
"""


_COOPMAT_MAIN = """\
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
        shared_b[i] = float16_t((n < pc.N && k < pc.K) ? {{WEIGHT_VALUE}}(n, k) : 0.0);
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
            output_values[m * pc.N + n] = shared_out[i];
        }
    }
}
"""


_Q4_K_COOPMAT_SOURCE = (
    _COOPMAT_COMMON_HEADER.replace("{{WEIGHT_BUFFER}}", "layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint weight[]; };\n")
    + """
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
"""
    + _COOPMAT_MAIN.replace("{{WEIGHT_VALUE}}", "q4k_value")
)


_Q8_0_COOPMAT_SOURCE = (
    _COOPMAT_COMMON_HEADER.replace("{{WEIGHT_BUFFER}}", "layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };\n")
    + """
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
"""
    + _COOPMAT_MAIN.replace("{{WEIGHT_VALUE}}", "q8_0_value")
)


Q4_K_COOPMAT_SOURCE = _Q4_K_COOPMAT_SOURCE
Q8_0_COOPMAT_SOURCE = _Q8_0_COOPMAT_SOURCE
