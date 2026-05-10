"""Q8_0 embedding shader for quantized OmniVoice."""

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
from torch2vk.vulkan.types import q8_0_halfwords_layout


OMNIVOICE_INPUT_EMBED_Q8_0_F32 = ShaderVariant(
    name="omnivoice_input_embed_q8_0_f32",
    family="quantized_omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceInputEmbedQ8_0Program",
        shader_name="omnivoice_input_embed_q8_0_f32",
        fields=(
            TensorFieldSpec(
                name="text_weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(
                    dtype="uint16",
                    shape=("TV", 544),
                    layout=q8_0_halfwords_layout(logical_k="H"),
                ),
            ),
            TensorFieldSpec(
                name="audio_weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(
                    dtype="uint16",
                    shape=("CV", 544),
                    layout=q8_0_halfwords_layout(logical_k="H"),
                ),
            ),
            TensorFieldSpec(
                name="batch_input_ids",
                io_kind=IOKind.INPUT,
                role="tokens",
                contract=TensorContract(dtype="int64", shape=("B", "C", "S")),
            ),
            TensorFieldSpec(
                name="batch_audio_mask",
                io_kind=IOKind.INPUT,
                role="mask",
                contract=TensorContract(dtype="uint32", shape=("B", "S")),
            ),
            TensorFieldSpec(
                name="hidden_states",
                io_kind=IOKind.OUTPUT,
                role="hidden_states",
                contract=TensorContract(dtype="float32", shape=("B", "S", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=20,
            fields=(
                PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B", dynamic=False),
                PushConstantFieldSpec("C", PushConstantType.UINT32, 4, "C", dynamic=False),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 8, "S", dynamic=False),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 12, "H", dynamic=False),
                PushConstantFieldSpec("V", PushConstantType.UINT32, 16, 1025, dynamic=False),
            ),
        ),
        dispatch=(ceil_div(mul(mul("B", "S"), "H"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        require_shader_int64=True,
        require_storage_buffer_16bit_access=True,
    ),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly TextWeightBuffer {
    uint16_t text_weight[];
};

layout(set = 0, binding = 1) buffer restrict readonly AudioWeightBuffer {
    uint16_t audio_weight[];
};

layout(set = 0, binding = 2) buffer restrict readonly BatchInputIdsBuffer {
    int64_t batch_input_ids[];
};

layout(set = 0, binding = 3) buffer restrict readonly BatchAudioMaskBuffer {
    uint batch_audio_mask[];
};

layout(set = 0, binding = 4) buffer restrict writeonly HiddenStatesBuffer {
    float hidden_states[];
};

layout(push_constant) uniform PushConstants {
    uint B;
    uint C;
    uint S;
    uint H;
    uint V;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

float q8_from_text(uint row, uint h) {
    const uint blocks_per_row = pc.H / 32u;
    const uint block_index = h >> 5u;
    const uint block_half = row * blocks_per_row * 17u + block_index * 17u;
    const float d = unpackHalf2x16(uint(text_weight[block_half])).x;
    const uint local = h & 31u;
    const uint packed = uint(text_weight[block_half + 1u + (local >> 1u)]);
    uint byte_value = ((local & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
    int quant = int(byte_value);
    if (quant >= 128) {
        quant -= 256;
    }
    return d * float(quant);
}

float q8_from_audio(uint row, uint h) {
    const uint blocks_per_row = pc.H / 32u;
    const uint block_index = h >> 5u;
    const uint block_half = row * blocks_per_row * 17u + block_index * 17u;
    const float d = unpackHalf2x16(uint(audio_weight[block_half])).x;
    const uint local = h & 31u;
    const uint packed = uint(audio_weight[block_half + 1u + (local >> 1u)]);
    uint byte_value = ((local & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
    int quant = int(byte_value);
    if (quant >= 128) {
        quant -= 256;
    }
    return d * float(quant);
}

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.S * pc.H;
    if (idx >= total) {
        return;
    }

    const uint h = idx % pc.H;
    const uint seq_idx = idx / pc.H;
    const uint s = seq_idx % pc.S;
    const uint b = seq_idx / pc.S;

    if (batch_audio_mask[b * pc.S + s] != 0u) {
        float value = 0.0;
        for (uint c = 0u; c < pc.C; ++c) {
            const uint input_offset = (b * pc.C + c) * pc.S + s;
            const uint token = uint(batch_input_ids[input_offset]);
            value += q8_from_audio(c * pc.V + token, h);
        }
        hidden_states[idx] = value;
        return;
    }

    const uint text_offset = (b * pc.C) * pc.S + s;
    const uint text_token = uint(batch_input_ids[text_offset]);
    hidden_states[idx] = q8_from_text(text_token, h);
}
""",
)
