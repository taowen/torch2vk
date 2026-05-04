"""Qwen3-ASR text prefill input embedding and audio scatter shader."""

from __future__ import annotations

from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderExecutionRequirements,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
)


QWEN3_ASR_TEXT_PREFILL_INPUTS_EMBEDS_F32 = ShaderVariant(
    name="qwen3_asr_text_prefill_inputs_embeds_f32",
    family="qwen3_asr.text",
    contract=ShaderContract(
        class_name="Qwen3AsrTextPrefillInputsEmbedsF32Program",
        shader_name="qwen3_asr_text_prefill_inputs_embeds_f32",
        fields=(
            TensorFieldSpec(
                name="input_ids",
                io_kind=IOKind.INPUT,
                role="input_ids",
                contract=TensorContract(dtype="int64", shape=(1, "T")),
            ),
            TensorFieldSpec(
                name="embed_tokens_weight",
                io_kind=IOKind.INPUT,
                role="embed_tokens_weight",
                contract=TensorContract(dtype="bfloat16", shape=("V", "H")),
            ),
            TensorFieldSpec(
                name="audio_features",
                io_kind=IOKind.INPUT,
                role="audio_features",
                contract=TensorContract(dtype="float32", shape=("A", "H")),
            ),
            TensorFieldSpec(
                name="audio_scatter_mask",
                io_kind=IOKind.OUTPUT,
                role="audio_scatter_mask",
                contract=TensorContract(dtype="uint32", shape=(1, "T", "H")),
            ),
            TensorFieldSpec(
                name="inputs_embeds",
                io_kind=IOKind.OUTPUT,
                role="inputs_embeds",
                contract=TensorContract(dtype="float32", shape=(1, "T", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=20,
            fields=(
                PushConstantFieldSpec("T", PushConstantType.UINT32, 0, "T"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
                PushConstantFieldSpec("A", PushConstantType.UINT32, 8, "A"),
                PushConstantFieldSpec("V", PushConstantType.UINT32, 12, "V"),
                PushConstantFieldSpec("audio_token_id", PushConstantType.UINT32, 16, 151676),
            ),
        ),
        dispatch=("T", 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        require_shader_int64=True,
        require_storage_buffer_16bit_access=True,
    ),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly InputIdsBuffer {
    int64_t input_ids[];
};

layout(set = 0, binding = 1) buffer restrict readonly EmbedTokensWeightBuffer {
    uint16_t embed_tokens_weight[];
};

layout(set = 0, binding = 2) buffer restrict readonly AudioFeaturesBuffer {
    float audio_features[];
};

layout(set = 0, binding = 3) buffer restrict writeonly AudioScatterMaskBuffer {
    uint audio_scatter_mask[];
};

layout(set = 0, binding = 4) buffer restrict writeonly InputsEmbedsBuffer {
    float inputs_embeds[];
};

layout(push_constant) uniform PushConstants {
    uint T;
    uint H;
    uint A;
    uint V;
    uint audio_token_id;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared uint shared_audio_row;

float bf16_to_f32(uint16_t value) {
    return uintBitsToFloat(uint(value) << 16);
}

bool is_audio_token(int64_t token_id) {
    return token_id == int64_t(pc.audio_token_id);
}

uint audio_row_before(uint token) {
    uint row = 0u;
    for (uint i = 0u; i < token; ++i) {
        if (is_audio_token(input_ids[i])) {
            row += 1u;
        }
    }
    return row;
}

void main() {
    const uint token = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    if (token >= pc.T) {
        return;
    }

    const int64_t token_id = input_ids[token];
    const bool audio_token = is_audio_token(token_id);
    if (tid == 0u) {
        shared_audio_row = audio_token ? audio_row_before(token) : 0u;
    }
    barrier();

    for (uint hidden = tid; hidden < pc.H; hidden += gl_WorkGroupSize.x) {
        const uint index = token * pc.H + hidden;
        audio_scatter_mask[index] = audio_token ? 1u : 0u;

        if (audio_token) {
            const uint audio_row = shared_audio_row;
            inputs_embeds[index] = audio_row < pc.A ? audio_features[audio_row * pc.H + hidden] : 0.0;
        } else if (token_id >= int64_t(0) && token_id < int64_t(pc.V)) {
            inputs_embeds[index] = bf16_to_f32(embed_tokens_weight[uint(token_id) * pc.H + hidden]);
        } else {
            inputs_embeds[index] = 0.0;
        }
    }
}
""".lstrip(),
)
