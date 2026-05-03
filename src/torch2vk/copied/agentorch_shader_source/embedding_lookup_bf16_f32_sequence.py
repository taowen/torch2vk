"""BF16 embedding row lookup into F32 sequence outputs."""

from __future__ import annotations

from agentorch.kernel.contract import (
    ceil_div,
    input_tensor,
    output_tensor,
    shader_contract,
    storage_buffer_binding,
    uniform_buffer_binding,
    uniform_ivec4,
)

from .shader_variant import shader_variant


EMBEDDING_LOOKUP_BF16_F32_SEQUENCE = shader_variant(
    name="embedding_lookup_bf16_f32_sequence",
    family="embedding_lookup_bf16_f32",
    contract=shader_contract(
        class_name="EmbeddingLookupBf16F32SequenceProgram",
        shader_name="embedding_lookup_bf16_f32_sequence",
        fields=(
            input_tensor(
                name="input_ids",
                binding="t_input_ids",
                role="input_ids",
                dtypes=("int32",),
                shape=("B", "S"),
            ),
            input_tensor(
                name="weight",
                binding="t_weight",
                role="weight",
                dtypes=("bfloat16",),
                shape=("V", "H"),
            ),
            output_tensor(
                name="output",
                binding="t_output",
                role="output",
                dtypes=("float32",),
                shape=("B", "S", "H"),
            ),
        ),
        uniforms=(uniform_ivec4(binding="sizes", value=("H", "S", "B", "V")),),
        dispatch=(ceil_div("H", 512), "S", "B"),
        bindings=(
            storage_buffer_binding(name="t_output", binding=0),
            storage_buffer_binding(name="t_input_ids", binding=1),
            storage_buffer_binding(name="t_weight", binding=2),
            uniform_buffer_binding(name="sizes", binding=3),
        ),
    ),
    source="""
#version 460

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
""".lstrip(),
)
