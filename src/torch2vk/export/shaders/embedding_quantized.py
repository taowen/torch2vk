from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source_for_shader,
    activation_glsl_type,
    activation_store,
    node_input_dtype,
    node_input_shape,
    node_output_shape,
    product_expr,
)
from torch2vk.export.shaders.embedding_f32 import make_embedding_variant
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
from torch2vk.vulkan.types import q4_k_words_layout, q8_0_halfwords_layout


_Q8_0_SOURCE = """\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
{{ACTIVATION_EXTENSION}}
{{INDEX_EXTENSION}}
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndicesBuffer { {{INDEX_TYPE}} indices[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint num_indices; uint embedding_dim; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

float q8_0_value(uint row, uint h) {
    const uint blocks_per_row = pc.embedding_dim / 32u;
    const uint block_index = h >> 5u;
    const uint block_half = row * blocks_per_row * 17u + block_index * 17u;
    const float d = unpackHalf2x16(uint(weight[block_half])).x;
    const uint local = h & 31u;
    const uint packed = uint(weight[block_half + 1u + (local >> 1u)]);
    uint byte_value = ((local & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
    int quant = int(byte_value);
    if (quant >= 128) {
        quant -= 256;
    }
    return d * float(quant);
}

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.num_indices * pc.embedding_dim) return;
    const uint token_idx = idx / pc.embedding_dim;
    const uint dim_idx = idx - token_idx * pc.embedding_dim;
    const {{INDEX_TYPE}} token_id = indices[token_idx];
    output_values[idx] = {{STORE_EMBEDDING}};
}
"""

_Q4_K_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}
{{INDEX_EXTENSION}}
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly WeightBuffer { uint weight[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndicesBuffer { {{INDEX_TYPE}} indices[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint num_indices; uint embedding_dim; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uint q4k_byte(uint block_word, uint byte_offset) {
    const uint word_value = weight[block_word + (byte_offset >> 2u)];
    return (word_value >> ((byte_offset & 3u) * 8u)) & 255u;
}

void q4k_scale_min(uint block_word, uint subblock, out uint scale, out uint minimum) {
    const uint scidx0 = (subblock < 4u) ? subblock : (subblock + 4u);
    const uint scidx1 = (subblock < 4u) ? subblock : (subblock - 4u);
    const uint scidxmask1 = (subblock < 4u) ? 0x30u : 0xC0u;
    const uint scidxshift1 = (subblock < 4u) ? 0u : 2u;
    const uint mbidx0 = subblock + 4u;
    const uint mbidx1 = (subblock < 4u) ? subblock + 4u : subblock;
    const uint mbidxmask0 = (subblock < 4u) ? 0x0Fu : 0xF0u;
    const uint mbidxshift0 = (subblock < 4u) ? 0u : 4u;
    const uint mbidxmask1 = (subblock < 4u) ? 0x30u : 0xC0u;
    const uint mbidxshift1 = (subblock < 4u) ? 0u : 2u;
    scale = (q4k_byte(block_word, 4u + scidx0) & 0x0Fu) |
        ((q4k_byte(block_word, 4u + scidx1) & scidxmask1) >> scidxshift1);
    minimum = ((q4k_byte(block_word, 4u + mbidx0) & mbidxmask0) >> mbidxshift0) |
        ((q4k_byte(block_word, 4u + mbidx1) & mbidxmask1) >> mbidxshift1);
}

float q4_k_value(uint row, uint h) {
    const uint blocks_per_row = pc.embedding_dim / 256u;
    const uint block_index = h >> 8u;
    const uint block_word = row * blocks_per_row * 36u + block_index * 36u;
    const uint local = h & 255u;
    const uint subblock = local >> 5u;
    uint scale;
    uint minimum;
    q4k_scale_min(block_word, subblock, scale, minimum);
    const vec2 dm = unpackHalf2x16(weight[block_word]);
    const uint q_byte_offset = ((local >> 6u) * 32u) + (local & 31u);
    const uint q_byte = q4k_byte(block_word, 16u + q_byte_offset);
    const uint q = ((local & 32u) == 0u) ? (q_byte & 15u) : (q_byte >> 4u);
    return dm.x * float(scale) * float(q) - dm.y * float(minimum);
}

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.num_indices * pc.embedding_dim) return;
    const uint token_idx = idx / pc.embedding_dim;
    const uint dim_idx = idx - token_idx * pc.embedding_dim;
    const {{INDEX_TYPE}} token_id = indices[token_idx];
    output_values[idx] = {{STORE_EMBEDDING}};
}
"""


def make_embedding_q4_k_m_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    weight_shape = node_input_shape(node, 0)
    indices_shape = node_input_shape(node, 1)
    if not out_shape or not weight_shape or not indices_shape:
        return None
    indices_dtype = node_input_dtype(node, 1)
    if indices_dtype not in {"int32", "int64"}:
        return None
    embedding_dim = int(weight_shape[-1])
    if embedding_dim % 256 != 0:
        return make_embedding_q8_0_variant(node, activation_dtype)

    idx_contract = tuple(f"I{i}" for i in range(len(indices_shape)))
    out_contract = idx_contract + ("H",)
    num_indices = product_expr(idx_contract)
    return ShaderVariant(
        name="embedding_q4_k_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportEmbeddingQ4_KProgram",
            shader_name="embedding_q4_k_f32",
            fields=(
                TensorFieldSpec(
                    "weight",
                    IOKind.INPUT,
                    "weight",
                    TensorContract(
                        dtype="uint32",
                        shape=("V", 36 * (embedding_dim // 256)),
                        layout=q4_k_words_layout(logical_k="H"),
                    ),
                ),
                TensorFieldSpec("indices", IOKind.INPUT, "input", TensorContract(dtype=indices_dtype, shape=idx_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype=activation_dtype, shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=8,
                fields=(
                    PushConstantFieldSpec("num_indices", PushConstantType.UINT32, 0, num_indices),
                    PushConstantFieldSpec("embedding_dim", PushConstantType.UINT32, 4, "H"),
                ),
            ),
            dispatch=(ceil_div(mul(num_indices, "H"), 256), 1, 1),
        ),
        execution_requirements=_execution_requirements(indices_dtype, activation_dtype),
        source=_q4_k_source(indices_dtype, activation_dtype),
    )


def make_embedding_q8_0_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    weight_shape = node_input_shape(node, 0)
    indices_shape = node_input_shape(node, 1)
    if not out_shape or not weight_shape or not indices_shape:
        return None
    indices_dtype = node_input_dtype(node, 1)
    if indices_dtype not in {"int32", "int64"}:
        return None
    embedding_dim = int(weight_shape[-1])
    if embedding_dim % 32 != 0:
        return make_embedding_variant(node, activation_dtype)

    idx_contract = tuple(f"I{i}" for i in range(len(indices_shape)))
    out_contract = idx_contract + ("H",)
    num_indices = product_expr(idx_contract)
    return ShaderVariant(
        name="embedding_q8_0_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportEmbeddingQ8_0Program",
            shader_name="embedding_q8_0_f32",
            fields=(
                TensorFieldSpec(
                    "weight",
                    IOKind.INPUT,
                    "weight",
                    TensorContract(
                        dtype="uint16",
                        shape=("V", 17 * (embedding_dim // 32)),
                        layout=q8_0_halfwords_layout(logical_k="H"),
                    ),
                ),
                TensorFieldSpec("indices", IOKind.INPUT, "input", TensorContract(dtype=indices_dtype, shape=idx_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype=activation_dtype, shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=8,
                fields=(
                    PushConstantFieldSpec("num_indices", PushConstantType.UINT32, 0, num_indices),
                    PushConstantFieldSpec("embedding_dim", PushConstantType.UINT32, 4, "H"),
                ),
            ),
            dispatch=(ceil_div(mul(num_indices, "H"), 256), 1, 1),
        ),
        execution_requirements=_execution_requirements(indices_dtype, activation_dtype),
        source=_q8_0_source(indices_dtype, activation_dtype),
    )


def _q4_k_source(dtype: str, activation_dtype: str) -> str:
    return _render_source(_Q4_K_SOURCE, "q4_k_value(uint(token_id), dim_idx)", dtype, activation_dtype)


def _q8_0_source(dtype: str, activation_dtype: str) -> str:
    return _render_source(_Q8_0_SOURCE, "q8_0_value(uint(token_id), dim_idx)", dtype, activation_dtype)


def _render_source(source: str, value: str, dtype: str, activation_dtype: str) -> str:
    index_type = "int64_t" if dtype == "int64" else "int"
    extension = "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require" if dtype == "int64" else ""
    return (
        source
        .replace("{{ACTIVATION_EXTENSION}}", activation_extension_source_for_shader(source, activation_dtype).rstrip())
        .replace("{{INDEX_EXTENSION}}", extension)
        .replace("{{INDEX_TYPE}}", index_type)
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_EMBEDDING}}", activation_store(value, activation_dtype))
    )


def _execution_requirements(dtype: str, activation_dtype: str) -> ShaderExecutionRequirements:
    return ShaderExecutionRequirements(
        require_shader_int64=dtype == "int64",
        require_storage_buffer_16bit_access=True,
    )
