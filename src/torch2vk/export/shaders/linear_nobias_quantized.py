from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source_for_shader,
    activation_glsl_type,
    activation_requirements,
    activation_store,
    activation_variant_name,
    node_input_shape,
    node_output_shape,
    product_expr,
    render_shader_template,
)
from torch2vk.export.shaders.linear_nobias_f32 import make_linear_nobias_variant
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
from torch2vk.vulkan.types import q4_k_words_layout, q6_k_halfwords_layout, q8_0_halfwords_layout


def make_linear_nobias_q4_k_m_variant(
    node: Node, activation_dtype: str = "float16"
) -> ShaderVariant | None:
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
                execution_requirements=_SUBGROUP64_REQUIREMENTS,
                activation_dtype=activation_dtype,
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
            execution_requirements=_coopmat_requirements(activation_dtype),
            activation_dtype=activation_dtype,
        )
    if k % 32 == 0:
        if m <= 8:
            return _make_q8_0_matvec_variant(
                x_shape=x_shape,
                w_shape=w_shape,
                out_shape=out_shape,
                activation_dtype=activation_dtype,
            )
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
            execution_requirements=_COOPMAT_REQUIREMENTS,
            activation_dtype=activation_dtype,
        )
    return make_linear_nobias_variant(node, activation_dtype)


def make_linear_nobias_q6_k_variant(
    node: Node,
    activation_dtype: str = "float16",
    *,
    matvec: bool | None = None,
) -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    w_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not x_shape or not w_shape or not out_shape:
        return None
    k = int(x_shape[-1])
    if k % 256 != 0:
        return None
    m = _flattened_rows(x_shape)
    use_matvec = (m <= 8) if matvec is None else matvec
    if use_matvec:
        return _make_quantized_variant(
            name="linear_nobias_q6_k_matvec_f32",
            class_name="ExportLinearNobiasQ6KMatvecProgram",
            x_shape=x_shape,
            w_shape=w_shape,
            out_shape=out_shape,
            weight_contract=TensorContract(
                dtype="uint16",
                shape=("N", mul(ceil_div("K", 256), 105)),
                layout=q6_k_halfwords_layout(logical_k="K"),
            ),
            source=_Q6_K_MATVEC_SOURCE,
            execution_requirements=_SUBGROUP64_16BIT_REQUIREMENTS,
            activation_dtype=activation_dtype,
        )
    return _make_quantized_variant(
        name="linear_nobias_q6_k_f32",
        class_name="ExportLinearNobiasQ6KProgram",
        x_shape=x_shape,
        w_shape=w_shape,
        out_shape=out_shape,
        weight_contract=TensorContract(
            dtype="uint16",
            shape=("N", mul(ceil_div("K", 256), 105)),
            layout=q6_k_halfwords_layout(logical_k="K"),
        ),
        source=_Q6_K_COOPMAT_SOURCE,
        execution_requirements=_coopmat_requirements(activation_dtype),
        activation_dtype=activation_dtype,
    )


def make_linear_nobias_q8_0_variant(
    node: Node, activation_dtype: str = "float16"
) -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    w_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not x_shape or not w_shape or not out_shape:
        return None
    k = int(x_shape[-1])
    if k % 32 != 0:
        return make_linear_nobias_variant(node, activation_dtype)
    m = _flattened_rows(x_shape)
    if m <= 8:
        return _make_q8_0_matvec_variant(
            x_shape=x_shape,
            w_shape=w_shape,
            out_shape=out_shape,
            activation_dtype=activation_dtype,
        )
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
        execution_requirements=_COOPMAT_REQUIREMENTS,
        activation_dtype=activation_dtype,
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
    execution_requirements: ShaderExecutionRequirements | None = None,
    activation_dtype: str = "float16",
) -> ShaderVariant:
    x_contract = tuple(f"X{i}" for i in range(len(x_shape) - 1)) + ("K",)
    out_contract = tuple(f"X{i}" for i in range(len(out_shape) - 1)) + ("N",)
    m = product_expr(tuple(f"X{i}" for i in range(len(x_shape) - 1)))
    dispatch = (
        (ceil_div("N", 2), m, 1)
        if "matvec" in name
        else (ceil_div("N", 64), ceil_div(m, 64), 1)
        if name in ("linear_nobias_q4_k_f32", "linear_nobias_q6_k_f32")
        else (ceil_div(m, 32), ceil_div("N", 16), 1)
    )
    return ShaderVariant(
        name=activation_variant_name(name, activation_dtype),
        family="export",
        contract=ShaderContract(
            class_name=class_name,
            shader_name=activation_variant_name(name, activation_dtype),
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=x_contract),
                ),
                TensorFieldSpec("weight", IOKind.INPUT, "weight", weight_contract),
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
                    PushConstantFieldSpec("K", PushConstantType.UINT32, 4, "K"),
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 8, "N"),
                ),
            ),
            dispatch=dispatch,
        ),
        execution_requirements=activation_requirements(activation_dtype, execution_requirements),
        source=_source(source, activation_dtype),
    )


def _make_q8_0_matvec_variant(
    *,
    x_shape: tuple[int, ...],
    w_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    activation_dtype: str,
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
        execution_requirements=_SUBGROUP64_16BIT_REQUIREMENTS,
        activation_dtype=activation_dtype,
    )


def _flattened_rows(x_shape: tuple[int, ...]) -> int:
    m = 1
    for dim in x_shape[:-1]:
        m *= int(dim)
    return m


def _source(source: str, activation_dtype: str) -> str:
    return render_shader_template(
        source,
        {
            "ACTIVATION_EXTENSION": activation_extension_source_for_shader(
                source, activation_dtype
            ),
            "ACTIVATION_TYPE": activation_glsl_type(activation_dtype),
            "ACTIVATION_VEC4_TYPE": _activation_vec4_glsl_type(activation_dtype),
            "ACCUM_TYPE": _accumulator_glsl_type(activation_dtype),
            "ACCUM_ZERO": _accumulator_zero(activation_dtype),
            "MATVEC_PAIR_X_VALUE": "float(x[row * pc.K + k_base + pair * 64u])",
            "MATVEC_X_VALUE": "float(x[row * pc.K + k])",
            "LOAD_A0": "float(x[m0 * pc.K + k])",
            "LOAD_A1": "float(x[m1 * pc.K + k])",
            "STORE_ACC0": activation_store("acc0", activation_dtype),
            "STORE_ACC1": activation_store("acc1", activation_dtype),
            "STORE_OUT0": activation_store("shared_out0[i]", activation_dtype),
            "STORE_OUT1": activation_store("shared_out1[i]", activation_dtype),
            "STORE_STAGE": activation_store("shared_stage[stage_index]", activation_dtype),
        },
    )


def _activation_vec4_glsl_type(dtype: str) -> str:
    if dtype == "float16":
        return "f16vec4"
    if dtype == "float32":
        return "vec4"
    raise ValueError(f"Unsupported activation dtype for shader generation: {dtype}")


def _accumulator_glsl_type(dtype: str) -> str:
    if dtype == "float16":
        return "float16_t"
    if dtype == "float32":
        return "float"
    raise ValueError(f"Unsupported activation dtype for shader generation: {dtype}")


def _accumulator_zero(dtype: str) -> str:
    if dtype == "float16":
        return "float16_t(0.0)"
    if dtype == "float32":
        return "0.0"
    raise ValueError(f"Unsupported activation dtype for shader generation: {dtype}")


def _coopmat_requirements(activation_dtype: str) -> ShaderExecutionRequirements:
    if activation_dtype == "float32":
        return _COOPMAT_REQUIREMENTS
    if activation_dtype == "float16":
        return _COOPMAT_F16ACC_REQUIREMENTS
    raise ValueError(f"Unsupported activation dtype for shader generation: {activation_dtype}")


_SUBGROUP64_REQUIREMENTS = ShaderExecutionRequirements(
    subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
)
_SUBGROUP64_16BIT_REQUIREMENTS = ShaderExecutionRequirements(
    subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
    require_storage_buffer_16bit_access=True,
)
_COOPMAT_REQUIREMENTS = ShaderExecutionRequirements(
    subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
    cooperative_matrix=CooperativeMatrixRequirements(
        scope="subgroup",
        m_size=16,
        n_size=16,
        k_size=16,
        a_type="float16",
        b_type="float16",
        c_type="float32",
        result_type="float32",
    ),
    require_storage_buffer_16bit_access=True,
)
_COOPMAT_F16ACC_REQUIREMENTS = ShaderExecutionRequirements(
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
)


_Q4_K_MATVEC_SOURCE = """\
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
{{ACTIVATION_EXTENSION}}\

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 0) buffer restrict readonly XVec4Buffer { {{ACTIVATION_VEC4_TYPE}} x4[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

uint q4k_byte(uint block_word, uint byte_offset) {
    const uint word_value = weight[block_word + (byte_offset >> 2u)];
    return (word_value >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

uint q4k_u16(uint block_word, uint half_index) {
    const uint byte_offset = 4u + half_index * 2u;
    return q4k_byte(block_word, byte_offset) | (q4k_byte(block_word, byte_offset + 1u) << 8u);
}

vec4 unpack8_f32(uint value) {
    return vec4(
        float(value & 255u),
        float((value >> 8u) & 255u),
        float((value >> 16u) & 255u),
        float((value >> 24u) & 255u)
    );
}

vec4 load_x4(uint row, uint k) {
    return vec4(x4[(row * pc.K + k) >> 2u]);
}

float q4k_block_dot(
    uint row,
    uint col,
    uint block_index,
    uint blocks_per_row,
    uint q_offset,
    uint y_offset,
    uint v_im
) {
    const uint block_word = col * blocks_per_row * 36u + block_index * 36u;
    const vec2 dm = unpackHalf2x16(weight[block_word]);

    const uint scale0_u32 = q4k_u16(block_word, v_im);
    const uint scale4_u32 = q4k_u16(block_word, v_im + 2u);
    const uint scale8_u32 = q4k_u16(block_word, v_im + 4u);
    const uint scale_0_4_l = (scale4_u32 << 16u) | scale0_u32;
    const uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2u;
    const vec4 scale_0_4_l_f = unpack8_f32(scale_0_4_l & 0x3F3F3F3Fu);
    const vec4 scale8_f = unpack8_f32((((scale8_u32 << 12u) | scale8_u32) & 0x0F0F0F0Fu) | scale_0_4_h);

    const float sc0 = scale_0_4_l_f.x;
    const float sc1 = scale_0_4_l_f.y;
    const float sc2 = scale_0_4_l_f.z;
    const float sc3 = scale_0_4_l_f.w;
    const float sc4 = scale8_f.x;
    const float sc5 = scale8_f.y;
    const float sc6 = scale8_f.z;
    const float sc7 = scale8_f.w;

    const uint qs0_u32 = weight[block_word + 4u + q_offset / 4u];
    const uint qs64_u32 = weight[block_word + 4u + q_offset / 4u + 16u];
    const vec4 qs0_lo4 = unpack8_f32(qs0_u32 & 0x0F0F0F0Fu);
    const vec4 qs64_lo4 = unpack8_f32(qs64_u32 & 0x0F0F0F0Fu);
    const vec4 qs0_hi4 = unpack8_f32((qs0_u32 >> 4u) & 0x0F0F0F0Fu);
    const vec4 qs64_hi4 = unpack8_f32((qs64_u32 >> 4u) & 0x0F0F0F0Fu);

    const uint y1_idx = block_index * 256u + y_offset;
    const uint y2_idx = y1_idx + 128u;
    const vec4 by10 = load_x4(row, y1_idx);
    const vec4 by132 = load_x4(row, y1_idx + 32u);
    const vec4 by20 = load_x4(row, y2_idx);
    const vec4 by232 = load_x4(row, y2_idx + 32u);

    const float sx = dot(by10, qs0_lo4);
    const float sy = dot(by132, qs0_hi4);
    const float sz = dot(by20, qs64_lo4);
    const float sw = dot(by232, qs64_hi4);
    const float smin =
        dot(by10, vec4(sc2)) +
        dot(by132, vec4(sc3)) +
        dot(by20, vec4(sc6)) +
        dot(by232, vec4(sc7));
    return dm.x * (sx * sc0 + sy * sc1 + sz * sc4 + sw * sc5) - dm.y * smin;
}

void main() {
    const uint lane = gl_SubgroupInvocationID;
    const uint row = gl_WorkGroupID.y;
    const uint col0 = gl_WorkGroupID.x * 2u;
    const uint col1 = col0 + 1u;
    float acc0 = 0.0;
    float acc1 = 0.0;
    if (row < pc.M) {
        const uint blocks_per_row = pc.K / 256u;
        const uint itid = lane & 15u;
        const uint ix = lane >> 4u;
        const uint il = itid >> 2u;
        const uint ir = itid & 3u;
        const uint v_im = il >> 1u;
        const uint v_in = il & 1u;
        const uint l0 = 4u * (2u * ir + v_in);
        const uint q_offset = 32u * v_im + l0;
        const uint y_offset = 64u * v_im + l0;
        for (uint block_index = ix; block_index < blocks_per_row; block_index += 4u) {
            if (col0 < pc.N) {
                acc0 += q4k_block_dot(row, col0, block_index, blocks_per_row, q_offset, y_offset, v_im);
            }
            if (col1 < pc.N) {
                acc1 += q4k_block_dot(row, col1, block_index, blocks_per_row, q_offset, y_offset, v_im);
            }
        }
    }
    acc0 = subgroupAdd(acc0);
    acc1 = subgroupAdd(acc1);
    if (lane == 0u && row < pc.M) {
        if (col0 < pc.N) { output_values[row * pc.N + col0] = {{STORE_ACC0}}; }
        if (col1 < pc.N) { output_values[row * pc.N + col1] = {{STORE_ACC1}}; }
    }
}
"""


_Q6_K_MATVEC_SOURCE = """\
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
{{ACTIVATION_EXTENSION}}\

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 0) buffer restrict readonly XVec4Buffer { {{ACTIVATION_VEC4_TYPE}} x4[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

uint q6k_byte(uint block_half, uint byte_offset) {
    const uint packed = uint(weight[block_half + (byte_offset >> 1u)]);
    return ((byte_offset & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
}

uint q6k_u32(uint block_half, uint byte_offset) {
    const uint half_index = block_half + (byte_offset >> 1u);
    return uint(weight[half_index]) | (uint(weight[half_index + 1u]) << 16u);
}

int q6k_i8(uint block_half, uint byte_offset) {
    int value = int(q6k_byte(block_half, byte_offset));
    if (value >= 128) {
        value -= 256;
    }
    return value;
}

vec4 unpack8_f32(uint value) {
    return vec4(
        float(value & 255u),
        float((value >> 8u) & 255u),
        float((value >> 16u) & 255u),
        float((value >> 24u) & 255u)
    );
}

vec4 load_x4(uint row, uint k) {
    return vec4(x4[(row * pc.K + k) >> 2u]);
}

float q6k_block_dot(uint row, uint col, uint block_index, uint itid) {
    const uint blocks_per_row = pc.K / 256u;
    const uint block_half = col * blocks_per_row * 105u + block_index * 105u;
    const uint k_base = block_index * 256u;

    const uint v_im = itid >> 3u;
    const uint v_in = itid - 8u * v_im;
    const uint l0 = 4u * v_in;
    const uint is = v_in >> 2u;
    const uint x_offset = k_base + 128u * v_im + l0;

    const uint ql_offset = 64u * v_im + l0;
    const uint qh_offset = 128u + 32u * v_im + l0;
    const uint scale_offset = 192u + 8u * v_im + is;

    const uint ql0 = q6k_u32(block_half, ql_offset);
    const uint ql32 = q6k_u32(block_half, ql_offset + 32u);
    const uint qh = q6k_u32(block_half, qh_offset);

    const uint q0_word = (ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4u);
    const uint q1_word = (ql32 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2u);
    const uint q2_word = ((ql0 >> 4u) & 0x0F0F0F0Fu) | (qh & 0x30303030u);
    const uint q3_word = ((ql32 >> 4u) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2u);

    const vec4 q0 = unpack8_f32(q0_word) - vec4(32.0);
    const vec4 q1 = unpack8_f32(q1_word) - vec4(32.0);
    const vec4 q2 = unpack8_f32(q2_word) - vec4(32.0);
    const vec4 q3 = unpack8_f32(q3_word) - vec4(32.0);

    const vec4 x0 = load_x4(row, x_offset);
    const vec4 x1 = load_x4(row, x_offset + 32u);
    const vec4 x2 = load_x4(row, x_offset + 64u);
    const vec4 x3 = load_x4(row, x_offset + 96u);

    const float d = unpackHalf2x16(uint(weight[block_half + 104u])).x;
    const float s0 = float(q6k_i8(block_half, scale_offset));
    const float s1 = float(q6k_i8(block_half, scale_offset + 2u));
    const float s2 = float(q6k_i8(block_half, scale_offset + 4u));
    const float s3 = float(q6k_i8(block_half, scale_offset + 6u));
    return d * (dot(q0, x0) * s0 + dot(q1, x1) * s1 + dot(q2, x2) * s2 + dot(q3, x3) * s3);
}

void main() {
    const uint col0 = gl_WorkGroupID.x * 2u;
    const uint col1 = col0 + 1u;
    const uint row = gl_WorkGroupID.y;
    const uint lane = gl_SubgroupInvocationID;
    if (row >= pc.M) {
        return;
    }

    const uint itid = lane & 15u;
    const uint ix = lane >> 4u;
    const uint blocks_per_row = pc.K / 256u;
    float acc0 = 0.0;
    float acc1 = 0.0;
    for (uint block = ix; block < blocks_per_row; block += 4u) {
        if (col0 < pc.N) {
            acc0 += q6k_block_dot(row, col0, block, itid);
        }
        if (col1 < pc.N) {
            acc1 += q6k_block_dot(row, col1, block, itid);
        }
    }

    acc0 = subgroupAdd(acc0);
    acc1 = subgroupAdd(acc1);
    if (lane == 0u) {
        if (col0 < pc.N) {
            output_values[row * pc.N + col0] = {{STORE_ACC0}};
        }
        if (col1 < pc.N) {
            output_values[row * pc.N + col1] = {{STORE_ACC1}};
        }
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
{{ACTIVATION_EXTENSION}}\

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };

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
            const float x_value = {{MATVEC_X_VALUE}};
            if (col0 < pc.N) { acc0 = fma(x_value, q8_0_value(col0, k), acc0); }
            if (col1 < pc.N) { acc1 = fma(x_value, q8_0_value(col1, k), acc1); }
        }
    }
    acc0 = subgroupAdd(acc0);
    acc1 = subgroupAdd(acc1);
    if (lane == 0u && row < pc.M) {
        if (col0 < pc.N) { output_values[row * pc.N + col0] = {{STORE_ACC0}}; }
        if (col1 < pc.N) { output_values[row * pc.N + col1] = {{STORE_ACC1}}; }
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

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
{{WEIGHT_BUFFER}}
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };

layout(push_constant) uniform PushConstants { uint M; uint K; uint N; } pc;

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
"""


_COOPMAT_MAIN = """\
void load_a_tile_pair(uint lane, uint row_base, uint k_base) {
    for (uint i = lane; i < TILE_SIZE; i += 64u) {
        const uint row = i / TILE_K;
        const uint col = i - row * TILE_K;
        const uint m0 = row_base + row;
        const uint m1 = row_base + TILE_M + row;
        const uint k = k_base + col;
        shared_a0[i] = float16_t((m0 < pc.M && k < pc.K) ? {{LOAD_A0}} : 0.0);
        shared_a1[i] = float16_t((m1 < pc.M && k < pc.K) ? {{LOAD_A1}} : 0.0);
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
    const uint row_base = gl_WorkGroupID.x * GROUP_M;
    const uint col_base = gl_WorkGroupID.y * TILE_N;

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_a0;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_a1;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_b;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> mat_c0;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> mat_c1;
    mat_c0 = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    mat_c1 = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint k_base = 0u; k_base < pc.K; k_base += TILE_K) {
        load_a_tile_pair(lane, row_base, k_base);
        load_b_tile(lane, col_base, k_base);
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
        if (n < pc.N) {
            if (m0 < pc.M) {
                output_values[m0 * pc.N + n] = {{STORE_OUT0}};
            }
            if (m1 < pc.M) {
                output_values[m1 * pc.N + n] = {{STORE_OUT1}};
            }
        }
    }
}
"""


_Q4_K_COOPMAT_SOURCE = """\
#version 460

#pragma use_vulkan_memory_model

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_cooperative_matrix : require
{{ACTIVATION_EXTENSION}}\

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 0) buffer restrict readonly XVec4Buffer { {{ACTIVATION_VEC4_TYPE}} x4[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };

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

shared f16vec2 shared_w[TILE_N * SHMEM_STRIDE];
shared f16vec2 shared_x[TILE_M * SHMEM_STRIDE];
shared {{ACCUM_TYPE}} shared_stage[WARPS * STAGE_SIZE];
shared float shared_q4_d[TILE_N];
shared float shared_q4_m[TILE_N];

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

uint q4k_quant4(uint block_word, uint local_k) {
    const uint pair = local_k >> 6u;
    const uint byte_index = local_k & 31u;
    const uint packed_q = weight[block_word + ((16u + pair * 32u + byte_index) >> 2u)];
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
            const vec2 dm = unpackHalf2x16(weight[block_word]);
            uint scale;
            uint minimum;
            q4k_scale_min(block_word, subblock, scale, minimum);
            shared_q4_d[row] = dm.x * float(scale);
            shared_q4_m[row] = dm.y * float(minimum);
        } else {
            shared_q4_d[row] = 0.0;
            shared_q4_m[row] = 0.0;
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
            const uint packed_q = q4k_quant4(block_word, local_k);
            const uint q0 = packed_q & 15u;
            const uint q1 = (packed_q >> 8u) & 15u;
            const uint q2 = (packed_q >> 16u) & 15u;
            const uint q3 = (packed_q >> 24u) & 15u;
            shared_w[base] = f16vec2(
                float16_t(shared_q4_d[row] * float(q0) - shared_q4_m[row]),
                float16_t(shared_q4_d[row] * float(q1) - shared_q4_m[row])
            );
            shared_w[base + 1u] = f16vec2(
                float16_t(shared_q4_d[row] * float(q2) - shared_q4_m[row]),
                float16_t(shared_q4_d[row] * float(q3) - shared_q4_m[row])
            );
        } else {
            shared_w[base] = f16vec2(0.0);
            shared_w[base + 1u] = f16vec2(0.0);
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
            const {{ACTIVATION_VEC4_TYPE}} values0 = x4[input_base >> 2u];
            const {{ACTIVATION_VEC4_TYPE}} values1 = x4[(input_base >> 2u) + 1u];
            shared_x[base] = f16vec2(float16_t(float(values0.x)), float16_t(float(values0.y)));
            shared_x[base + 1u] = f16vec2(float16_t(float(values0.z)), float16_t(float(values0.w)));
            shared_x[base + 2u] = f16vec2(float16_t(float(values1.x)), float16_t(float(values1.y)));
            shared_x[base + 3u] = f16vec2(float16_t(float(values1.z)), float16_t(float(values1.w)));
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

void main() {
    const uint local_id = gl_LocalInvocationID.x;
    const uint lane = gl_SubgroupInvocationID;
    const uint warp = gl_SubgroupID;
    const uint n_base = gl_WorkGroupID.x * TILE_N;
    const uint m_base = gl_WorkGroupID.y * TILE_M;

    const uint store_r = lane % TM;
    const uint store_c = lane / TM;
    const uint store_stride = WARP / TM;

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_w;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_x;
    coopmat<{{ACCUM_TYPE}}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> sums[CMS_PER_ROW * CMS_PER_COL];

    [[unroll]] for (uint i = 0u; i < CMS_PER_ROW * CMS_PER_COL; ++i) {
        sums[i] = coopmat<{{ACCUM_TYPE}}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>({{ACCUM_ZERO}});
    }

    for (uint k_base = 0u; k_base < pc.K; k_base += 32u) {
        prepare_q4k_tile_scales(local_id, n_base, k_base);
        barrier();
        load_weight_tile(local_id, n_base, k_base);
        load_x_tile(local_id, m_base, k_base);
        barrier();

        [[unroll]] for (uint kk = 0u; kk < TILE_K; kk += TK) {
            [[unroll]] for (uint cm_row = 0u; cm_row < CMS_PER_ROW; ++cm_row) {
                coopMatLoad(mat_w, shared_w, int(cm_row * TM * SHMEM_STRIDE + kk / 2u), int(SHMEM_STRIDE), gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint cm_col = 0u; cm_col < CMS_PER_COL; ++cm_col) {
                    coopMatLoad(mat_x, shared_x, int((warp * WN + cm_col * TN) * SHMEM_STRIDE + kk / 2u), int(SHMEM_STRIDE), gl_CooperativeMatrixLayoutColumnMajor);
                    const uint sum_index = cm_col * CMS_PER_ROW + cm_row;
                    sums[sum_index] = coopMatMulAdd(mat_w, mat_x, sums[sum_index]);
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
            coopmat<{{ACCUM_TYPE}}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> output_tile =
                coopmat<{{ACCUM_TYPE}}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(sums[sum_index]);
            if (n_tile + TM <= pc.N && m_tile + TN <= pc.M) {
                coopMatStore(output_tile, output_values, int(m_tile * pc.N + n_tile), int(pc.N), gl_CooperativeMatrixLayoutColumnMajor);
            } else {
                coopMatStore(output_tile, shared_stage, int(warp * STAGE_SIZE), int(TM), gl_CooperativeMatrixLayoutColumnMajor);
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);

                [[unroll]] for (uint col = 0u; col < TN; col += store_stride) {
                    const uint n = n_tile + store_r;
                    const uint m = m_tile + col + store_c;
                    if (n < pc.N && m < pc.M) {
                        const uint stage_index = warp * STAGE_SIZE + (col + store_c) * TM + store_r;
                        output_values[m * pc.N + n] = {{STORE_STAGE}};
                    }
                }
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
            }
        }
    }
}
"""


_Q6_K_COOPMAT_SOURCE = """\
#version 460

#pragma use_vulkan_memory_model

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_cooperative_matrix : require
{{ACTIVATION_EXTENSION}}\

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 0) buffer restrict readonly XVec4Buffer { {{ACTIVATION_VEC4_TYPE}} x4[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };

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

shared f16vec2 shared_w[TILE_N * SHMEM_STRIDE];
shared f16vec2 shared_x[TILE_M * SHMEM_STRIDE];
shared {{ACCUM_TYPE}} shared_stage[WARPS * STAGE_SIZE];

uint q6k_byte(uint block_half, uint byte_offset) {
    const uint packed = uint(weight[block_half + (byte_offset >> 1u)]);
    return ((byte_offset & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
}

int q6k_i8(uint block_half, uint byte_offset) {
    int value = int(q6k_byte(block_half, byte_offset));
    if (value >= 128) {
        value -= 256;
    }
    return value;
}

f16vec2 q6k_pair(uint block_half, uint local_k) {
    const uint iqs = (local_k >> 1u) & 127u;
    const uint section = iqs >> 6u;
    const uint b = ((iqs & 63u) >> 5u) * 4u;
    const uint is_b = (iqs & 15u) >> 3u;
    const uint qhshift = ((iqs & 63u) >> 4u) * 2u;
    const uint scale_index = 8u * section + qhshift + is_b;
    const uint qsi = section * 32u + (iqs & 31u);
    const uint qhi = section * 16u + (iqs & 15u);
    const float d = unpackHalf2x16(uint(weight[block_half + 104u])).x;
    const float dscale = d * float(q6k_i8(block_half, 192u + scale_index));
    const uint ql = (uint(weight[block_half + qsi]) >> b) & 0x0F0Fu;
    const uint qh = (uint(weight[block_half + 64u + qhi]) >> qhshift) & 0x0303u;
    const uint packed = ql | (qh << 4u);
    return f16vec2(
        float16_t((float(int(packed & 255u)) - 32.0) * dscale),
        float16_t((float(int((packed >> 8u) & 255u)) - 32.0) * dscale)
    );
}

void load_weight_tile(uint local_id, uint n_base, uint k_base) {
    const uint blocks_per_row = pc.K / 256u;
    const uint block_index = k_base >> 8u;
    for (uint i = local_id; i < TILE_NK_LOADS; i += 128u) {
        const uint row = i / TILE_K_WEIGHT_LOADS;
        const uint k_offset = (i - row * TILE_K_WEIGHT_LOADS) * WEIGHT_VALUES_PER_LOAD;
        const uint n = n_base + row;
        const uint k = k_base + k_offset;
        const uint base = row * SHMEM_STRIDE + k_offset / 2u;
        if (n < pc.N && k < pc.K) {
            const uint block_half = n * blocks_per_row * 105u + block_index * 105u;
            const uint local_k = k & 255u;
            shared_w[base] = q6k_pair(block_half, local_k);
            shared_w[base + 1u] = q6k_pair(block_half, local_k + 2u);
        } else {
            shared_w[base] = f16vec2(0.0);
            shared_w[base + 1u] = f16vec2(0.0);
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
            const {{ACTIVATION_VEC4_TYPE}} values0 = x4[input_base >> 2u];
            const {{ACTIVATION_VEC4_TYPE}} values1 = x4[(input_base >> 2u) + 1u];
            shared_x[base] = f16vec2(float16_t(float(values0.x)), float16_t(float(values0.y)));
            shared_x[base + 1u] = f16vec2(float16_t(float(values0.z)), float16_t(float(values0.w)));
            shared_x[base + 2u] = f16vec2(float16_t(float(values1.x)), float16_t(float(values1.y)));
            shared_x[base + 3u] = f16vec2(float16_t(float(values1.z)), float16_t(float(values1.w)));
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

void main() {
    const uint local_id = gl_LocalInvocationID.x;
    const uint lane = gl_SubgroupInvocationID;
    const uint warp = gl_SubgroupID;
    const uint n_base = gl_WorkGroupID.x * TILE_N;
    const uint m_base = gl_WorkGroupID.y * TILE_M;

    const uint store_r = lane % TM;
    const uint store_c = lane / TM;
    const uint store_stride = WARP / TM;

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_w;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_x;
    coopmat<{{ACCUM_TYPE}}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> sums[CMS_PER_ROW * CMS_PER_COL];

    [[unroll]] for (uint i = 0u; i < CMS_PER_ROW * CMS_PER_COL; ++i) {
        sums[i] = coopmat<{{ACCUM_TYPE}}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>({{ACCUM_ZERO}});
    }

    for (uint k_base = 0u; k_base < pc.K; k_base += TILE_K) {
        load_weight_tile(local_id, n_base, k_base);
        load_x_tile(local_id, m_base, k_base);
        barrier();

        [[unroll]] for (uint kk = 0u; kk < TILE_K; kk += TK) {
            [[unroll]] for (uint cm_row = 0u; cm_row < CMS_PER_ROW; ++cm_row) {
                coopMatLoad(mat_w, shared_w, int(cm_row * TM * SHMEM_STRIDE + kk / 2u), int(SHMEM_STRIDE), gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint cm_col = 0u; cm_col < CMS_PER_COL; ++cm_col) {
                    coopMatLoad(mat_x, shared_x, int((warp * WN + cm_col * TN) * SHMEM_STRIDE + kk / 2u), int(SHMEM_STRIDE), gl_CooperativeMatrixLayoutColumnMajor);
                    const uint sum_index = cm_col * CMS_PER_ROW + cm_row;
                    sums[sum_index] = coopMatMulAdd(mat_w, mat_x, sums[sum_index]);
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
            coopmat<{{ACCUM_TYPE}}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> output_tile =
                coopmat<{{ACCUM_TYPE}}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(sums[sum_index]);
            if (n_tile + TM <= pc.N && m_tile + TN <= pc.M) {
                coopMatStore(output_tile, output_values, int(m_tile * pc.N + n_tile), int(pc.N), gl_CooperativeMatrixLayoutColumnMajor);
            } else {
                coopMatStore(output_tile, shared_stage, int(warp * STAGE_SIZE), int(TM), gl_CooperativeMatrixLayoutColumnMajor);
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);

                [[unroll]] for (uint col = 0u; col < TN; col += store_stride) {
                    const uint n = n_tile + store_r;
                    const uint m = m_tile + col + store_c;
                    if (n < pc.N && m < pc.M) {
                        const uint stage_index = warp * STAGE_SIZE + (col + store_c) * TM + store_r;
                        output_values[m * pc.N + n] = {{STORE_STAGE}};
                    }
                }
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
            }
        }
    }
}
"""


_Q8_0_COOPMAT_SOURCE = (
    _COOPMAT_COMMON_HEADER.replace(
        "{{WEIGHT_BUFFER}}",
        "layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };\n",
    )
    + """
shared float shared_q8_d[TILE_N];

void prepare_q8_0_tile_scales(uint lane, uint col_base, uint k_base) {
    const uint blocks_per_row = pc.K / 32u;
    const uint block_index = k_base >> 5u;
    for (uint col = lane; col < TILE_N; col += 64u) {
        const uint n = col_base + col;
        if (n < pc.N) {
            const uint block_half = n * blocks_per_row * 17u + block_index * 17u;
            shared_q8_d[col] = unpackHalf2x16(uint(weight[block_half])).x;
        } else {
            shared_q8_d[col] = 0.0;
        }
    }
}

float q8_0_value_prepared(uint row, uint k) {
    const uint blocks_per_row = pc.K / 32u;
    const uint block_index = k >> 5u;
    const uint block_half = row * blocks_per_row * 17u + block_index * 17u;
    const uint local = k & 31u;
    const uint packed = uint(weight[block_half + 1u + (local >> 1u)]);
    uint byte_value = ((local & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
    int quant = int(byte_value);
    if (quant >= 128) { quant -= 256; }
    return shared_q8_d[row & 15u] * float(quant);
}
"""
    + _COOPMAT_MAIN.replace(
        "void load_b_tile(uint lane, uint col_base, uint k_base) {\n"
        "    for (uint i = lane; i < TILE_SIZE; i += 64u) {\n"
        "        const uint col = i / TILE_K;\n"
        "        const uint k_offset = i - col * TILE_K;\n"
        "        const uint n = col_base + col;\n"
        "        const uint k = k_base + k_offset;\n"
        "        shared_b[i] = float16_t((n < pc.N && k < pc.K) ? {{WEIGHT_VALUE}}(n, k) : 0.0);\n"
        "    }\n"
        "}\n",
        "void load_b_tile(uint lane, uint col_base, uint k_base) {\n"
        "    for (uint i = lane; i < TILE_SIZE; i += 64u) {\n"
        "        const uint col = i / TILE_K;\n"
        "        const uint k_offset = i - col * TILE_K;\n"
        "        const uint n = col_base + col;\n"
        "        const uint k = k_base + k_offset;\n"
        "        shared_b[i] = float16_t((n < pc.N && k < pc.K) ? q8_0_value_prepared(n, k) : 0.0);\n"
        "    }\n"
        "}\n",
    ).replace(
        "    for (uint k_base = 0u; k_base < pc.K; k_base += TILE_K) {\n"
        "        load_a_tile_pair(lane, row_base, k_base);\n"
        "        load_b_tile(lane, col_base, k_base);\n"
        "        barrier();\n"
        "        coopMatLoad(mat_a0, shared_a0, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);\n"
        "        coopMatLoad(mat_a1, shared_a1, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);\n"
        "        coopMatLoad(mat_b, shared_b, 0, int(TILE_K), gl_CooperativeMatrixLayoutColumnMajor);\n"
        "        mat_c0 = coopMatMulAdd(mat_a0, mat_b, mat_c0);\n"
        "        mat_c1 = coopMatMulAdd(mat_a1, mat_b, mat_c1);\n"
        "        barrier();\n"
        "    }\n",
        "    for (uint k_base = 0u; k_base < pc.K; k_base += 32u) {\n"
        "        prepare_q8_0_tile_scales(lane, col_base, k_base);\n"
        "        barrier();\n"
        "\n"
        "        load_a_tile_pair(lane, row_base, k_base);\n"
        "        load_b_tile(lane, col_base, k_base);\n"
        "        barrier();\n"
        "        coopMatLoad(mat_a0, shared_a0, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);\n"
        "        coopMatLoad(mat_a1, shared_a1, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);\n"
        "        coopMatLoad(mat_b, shared_b, 0, int(TILE_K), gl_CooperativeMatrixLayoutColumnMajor);\n"
        "        mat_c0 = coopMatMulAdd(mat_a0, mat_b, mat_c0);\n"
        "        mat_c1 = coopMatMulAdd(mat_a1, mat_b, mat_c1);\n"
        "        barrier();\n"
        "\n"
        "        load_a_tile_pair(lane, row_base, k_base + TILE_K);\n"
        "        load_b_tile(lane, col_base, k_base + TILE_K);\n"
        "        barrier();\n"
        "        coopMatLoad(mat_a0, shared_a0, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);\n"
        "        coopMatLoad(mat_a1, shared_a1, 0, int(TILE_K), gl_CooperativeMatrixLayoutRowMajor);\n"
        "        coopMatLoad(mat_b, shared_b, 0, int(TILE_K), gl_CooperativeMatrixLayoutColumnMajor);\n"
        "        mat_c0 = coopMatMulAdd(mat_a0, mat_b, mat_c0);\n"
        "        mat_c1 = coopMatMulAdd(mat_a1, mat_b, mat_c1);\n"
        "        barrier();\n"
        "    }\n",
    )
)


Q4_K_COOPMAT_SOURCE = _Q4_K_COOPMAT_SOURCE
Q8_0_COOPMAT_SOURCE = _Q8_0_COOPMAT_SOURCE
