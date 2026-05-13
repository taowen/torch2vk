"""Q4_K embedding lookup with float32 activation output."""

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
from torch2vk.vulkan.types import q4_k_words_layout


EMBEDDING_Q4_K_ACT_F32 = ShaderVariant(
    name="embedding_q4_k_act_f32",
    family="optimized_qwen3",
    contract=ShaderContract(
        class_name="OptimizedQwen3EmbeddingQ4KActF32Program",
        shader_name="embedding_q4_k_act_f32",
        fields=(
            TensorFieldSpec(
                "weight",
                IOKind.INPUT,
                "weight",
                TensorContract(
                    dtype="uint32",
                    shape=("V", 144),
                    layout=q4_k_words_layout(logical_k="H"),
                ),
            ),
            TensorFieldSpec("indices", IOKind.INPUT, "input", TensorContract(dtype="int64", shape=("I0", "I1"))),
            TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=("I0", "I1", "H"))),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("num_indices", PushConstantType.UINT32, 0, mul("I0", "I1")),
                PushConstantFieldSpec("embedding_dim", PushConstantType.UINT32, 4, "H"),
            ),
        ),
        dispatch=(ceil_div(mul(mul("I0", "I1"), "H"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly WeightBuffer { uint weight[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndicesBuffer { int64_t indices[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
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
    const int64_t token_id = indices[token_idx];
    output_values[idx] = q4_k_value(uint(token_id), dim_idx);
}
""",
)
