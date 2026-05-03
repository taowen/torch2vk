"""Qwen3 embedding lookup shader."""

from __future__ import annotations

from torch2vk.shader import (
    Binding,
    BindingAccess,
    ShaderContract,
    ShaderVariant,
    TensorContract,
    UniformBlock,
)

_SOURCE = """#version 460

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer {
    float t_output[];
};

layout(set = 0, binding = 1) buffer restrict readonly InputIdsBuffer {
    int t_input_ids[];
};

layout(set = 0, binding = 2) buffer restrict readonly WeightBuffer {
    uint16_t t_weight[];
};

layout(set = 0, binding = 3) uniform restrict readonly sizes_UBO {
    ivec4 sizes;
};

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

float bf16_to_f32(const uint16_t bits) {
    return uintBitsToFloat(uint(bits) << 16);
}

void main() {
    const uint hidden_index = gl_GlobalInvocationID.x;
    const uint step_index = gl_GlobalInvocationID.y;
    const uint batch_index = gl_GlobalInvocationID.z;

    const uint hidden = uint(sizes.x);
    const uint steps = uint(sizes.y);
    const uint batches = uint(sizes.z);
    const uint vocab_size = uint(sizes.w);

    if (hidden_index >= hidden || step_index >= steps || batch_index >= batches) {
        return;
    }

    const uint token_offset = batch_index * steps + step_index;
    const int token_id = t_input_ids[token_offset];
    if (token_id < 0 || uint(token_id) >= vocab_size) {
        return;
    }

    t_output[token_offset * hidden + hidden_index] = bf16_to_f32(t_weight[uint(token_id) * hidden + hidden_index]);
}
"""

EMBEDDING_LOOKUP_BF16_F32 = ShaderVariant(
    name="embedding_lookup_bf16_f32_sequence",
    family="embedding_lookup",
    contract=ShaderContract(
        name="embedding_lookup_bf16_f32_sequence",
        inputs={
            "input_ids": TensorContract(dtype="int32", shape=("B", "S")),
            "weight": TensorContract(dtype="bfloat16", shape=("V", "H")),
        },
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "H"))},
        bindings=(
            Binding("output", 0, BindingAccess.WRITE),
            Binding("input_ids", 1, BindingAccess.READ),
            Binding("weight", 2, BindingAccess.READ),
        ),
        dispatch=("H", "S", "B"),
        uniforms=(UniformBlock("sizes", 3, ("H", "S", "B", "V")),),
    ),
    source=_SOURCE,
)
