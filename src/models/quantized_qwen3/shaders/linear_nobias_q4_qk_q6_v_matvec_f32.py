"""Fused Q4_K Q/K and Q6_K V decode matvec for Qwen3."""

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
    add,
    ceil_div,
    mul,
)
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements, SubgroupRequirements
from torch2vk.vulkan.types import q4_k_words_layout, q6_k_halfwords_layout

from models.quantized_qwen3.shaders.linear_nobias_q4_k_qk_matvec_f32 import (
    LINEAR_NOBIAS_Q4_K_QK_MATVEC_F32,
)
from models.quantized_qwen3.shaders.linear_nobias_q6_k_matvec_f32 import (
    LINEAR_NOBIAS_Q6_K_MATVEC_F32,
)


def _q6_helpers_source() -> str:
    source = LINEAR_NOBIAS_Q6_K_MATVEC_F32.source
    prefix_start = source.index("uint q6k_byte")
    prefix_end = source.index("vec4 unpack8_f32")
    block_start = source.index("float q6k_block_dot")
    block_end = source.index("void main()")
    helpers = source[prefix_start:prefix_end] + source[block_start:block_end]
    helpers = helpers.replace("weight[", "v_weight[")
    return helpers + """\

float q6k_dot_v(uint row, uint col) {
    const uint lane = gl_SubgroupInvocationID;
    const uint itid = lane & 15u;
    const uint ix = lane >> 4u;
    const uint blocks_per_row = pc.K / 256u;
    float acc = 0.0;
    for (uint block = ix; block < blocks_per_row; block += 4u) {
        acc += q6k_block_dot(row, col, block, itid);
    }
    return subgroupAdd(acc);
}
"""


def _source() -> str:
    source = LINEAR_NOBIAS_Q4_K_QK_MATVEC_F32.source
    source = source.replace(
        "layout(set = 0, binding = 3) buffer restrict writeonly QOutputBuffer { float16_t q_output[]; };\n"
        "layout(set = 0, binding = 4) buffer restrict writeonly KOutputBuffer { float16_t k_output[]; };",
        "\n".join(
            (
                "layout(set = 0, binding = 3) buffer restrict readonly VWeightBuffer { uint16_t v_weight[]; };",
                "layout(set = 0, binding = 4) buffer restrict writeonly QOutputBuffer { float16_t q_output[]; };",
                "layout(set = 0, binding = 5) buffer restrict writeonly KOutputBuffer { float16_t k_output[]; };",
                "layout(set = 0, binding = 6) buffer restrict writeonly VOutputBuffer { float16_t v_output[]; };",
            )
        ),
    )
    source = source.replace(
        "layout(push_constant) uniform PushConstants { uint M; uint K; uint NQ; uint NK; } pc;",
        "layout(push_constant) uniform PushConstants { uint M; uint K; uint NQ; uint NK; uint NV; } pc;",
    )
    source = source.replace("void store_output", _q6_helpers_source() + "\nvoid store_output")
    source = source.replace(
        "    if (k_col < pc.NK) {\n"
        "        k_output[row * pc.NK + k_col] = float16_t(value);\n"
        "    }\n",
        "    if (k_col < pc.NK) {\n"
        "        k_output[row * pc.NK + k_col] = float16_t(value);\n"
        "        return;\n"
        "    }\n"
        "    const uint v_col = k_col - pc.NK;\n"
        "    if (v_col < pc.NV) {\n"
        "        v_output[row * pc.NV + v_col] = float16_t(value);\n"
        "    }\n",
    )
    source = source.replace(
        "    const uint total_n = pc.NQ + pc.NK;",
        "    const uint total_n = pc.NQ + pc.NK + pc.NV;",
    )
    source = source.replace(
        "    const float acc = col < pc.NQ ? q4k_dot_q(row, col) : q4k_dot_k(row, col - pc.NQ);",
        "    float acc;\n"
        "    if (col < pc.NQ) {\n"
        "        acc = q4k_dot_q(row, col);\n"
        "    } else if (col < pc.NQ + pc.NK) {\n"
        "        acc = q4k_dot_k(row, col - pc.NQ);\n"
        "    } else {\n"
        "        acc = q6k_dot_v(row, col - pc.NQ - pc.NK);\n"
        "    }",
    )
    return source


LINEAR_NOBIAS_Q4_QK_Q6_V_MATVEC_F32 = ShaderVariant(
    name="linear_nobias_q4_qk_q6_v_matvec_f32",
    family="quantized_qwen3",
    contract=ShaderContract(
        class_name="LinearNobiasQ4QKQ6VMatvecProgram",
        shader_name="linear_nobias_q4_qk_q6_v_matvec_f32",
        fields=(
            TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float16", shape=("X0", "X1", "K"))),
            TensorFieldSpec(
                "q_weight",
                IOKind.INPUT,
                "weight",
                TensorContract(
                    dtype="uint32",
                    shape=("NQ", mul(ceil_div("K", 256), 36)),
                    layout=q4_k_words_layout(logical_k="K"),
                ),
            ),
            TensorFieldSpec(
                "k_weight",
                IOKind.INPUT,
                "weight",
                TensorContract(
                    dtype="uint32",
                    shape=("NK", mul(ceil_div("K", 256), 36)),
                    layout=q4_k_words_layout(logical_k="K"),
                ),
            ),
            TensorFieldSpec(
                "v_weight",
                IOKind.INPUT,
                "weight",
                TensorContract(
                    dtype="uint16",
                    shape=("NV", mul(ceil_div("K", 256), 105)),
                    layout=q6_k_halfwords_layout(logical_k="K"),
                ),
            ),
            TensorFieldSpec("q_output", IOKind.OUTPUT, "output", TensorContract(dtype="float16", shape=("X0", "X1", "NQ"))),
            TensorFieldSpec("k_output", IOKind.OUTPUT, "output", TensorContract(dtype="float16", shape=("X0", "X1", "NK"))),
            TensorFieldSpec("v_output", IOKind.OUTPUT, "output", TensorContract(dtype="float16", shape=("X0", "X1", "NV"))),
        ),
        push_constants=PushConstantSpec(
            size=20,
            fields=(
                PushConstantFieldSpec("M", PushConstantType.UINT32, 0, mul("X0", "X1")),
                PushConstantFieldSpec("K", PushConstantType.UINT32, 4, "K"),
                PushConstantFieldSpec("NQ", PushConstantType.UINT32, 8, "NQ"),
                PushConstantFieldSpec("NK", PushConstantType.UINT32, 12, "NK"),
                PushConstantFieldSpec("NV", PushConstantType.UINT32, 16, "NV"),
            ),
        ),
        dispatch=(ceil_div(add(add("NQ", "NK"), "NV"), 2), mul("X0", "X1"), 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        require_storage_buffer_16bit_access=True,
    ),
    source=_source(),
)
