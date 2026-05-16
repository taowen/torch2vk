"""OmniVoice model-level Vulkan shader variants."""

from __future__ import annotations

from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantInput,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderExecutionRequirements,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
    ceil_div,
    mul,
)


_AUDIO_VOCAB_SIZE = 1025


OMNIVOICE_INPUT_EMBED_F32 = ShaderVariant(
    name="omnivoice_input_embed_f32",
    family="omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceInputEmbedF32Program",
        shader_name="omnivoice_input_embed_f32",
        fields=(
            TensorFieldSpec(
                "text_weight",
                IOKind.INPUT,
                "weight",
                TensorContract(dtype="float32", shape=("TV", "H")),
            ),
            TensorFieldSpec(
                "audio_weight",
                IOKind.INPUT,
                "weight",
                TensorContract(dtype="float32", shape=("CV", "H")),
            ),
            TensorFieldSpec(
                "batch_input_ids",
                IOKind.INPUT,
                "tokens",
                TensorContract(dtype="int64", shape=("B", "C", "S")),
            ),
            TensorFieldSpec(
                "batch_audio_mask",
                IOKind.INPUT,
                "mask",
                TensorContract(dtype="uint32", shape=("B", "S")),
            ),
            TensorFieldSpec(
                "hidden_states",
                IOKind.OUTPUT,
                "hidden_states",
                TensorContract(dtype="float16", shape=("B", "S", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=20,
            fields=(
                PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                PushConstantFieldSpec("C", PushConstantType.UINT32, 4, "C"),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 8, "S"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 12, "H"),
                PushConstantFieldSpec("V", PushConstantType.UINT32, 16, _AUDIO_VOCAB_SIZE),
            ),
        ),
        dispatch=(ceil_div(mul(mul("B", "S"), "H"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        require_shader_int64=True,
        require_storage_buffer_16bit_access=True,
    ),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly TextWeightBuffer {
    float text_weight[];
};

layout(set = 0, binding = 1) buffer restrict readonly AudioWeightBuffer {
    float audio_weight[];
};

layout(set = 0, binding = 2) buffer restrict readonly BatchInputIdsBuffer {
    int64_t batch_input_ids[];
};

layout(set = 0, binding = 3) buffer restrict readonly BatchAudioMaskBuffer {
    uint batch_audio_mask[];
};

layout(set = 0, binding = 4) buffer restrict writeonly HiddenStatesBuffer {
    float16_t hidden_states[];
};

layout(push_constant) uniform PushConstants {
    uint B;
    uint C;
    uint S;
    uint H;
    uint V;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

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
            const uint row = c * pc.V + token;
            value += float(audio_weight[row * pc.H + h]);
        }
        hidden_states[idx] = float16_t(value);
        return;
    }

    const uint text_offset = (b * pc.C) * pc.S + s;
    const uint text_token = uint(batch_input_ids[text_offset]);
    hidden_states[idx] = float16_t(text_weight[text_token * pc.H + h]);
}
""".lstrip(),
)


OMNIVOICE_CFG_SCORE_F32 = ShaderVariant(
    name="omnivoice_cfg_score_f32",
    family="omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceCfgScoreF32Program",
        shader_name="omnivoice_cfg_score_f32",
        fields=(
            TensorFieldSpec(
                "logits",
                IOKind.INPUT,
                "logits",
                TensorContract(dtype="float16", shape=(2, "S", "CV")),
            ),
            TensorFieldSpec(
                "tokens",
                IOKind.INPUT,
                "tokens",
                TensorContract(dtype="int64", shape=(1, "C", "T")),
            ),
            TensorFieldSpec(
                "candidate_tokens",
                IOKind.OUTPUT,
                "candidate_tokens",
                TensorContract(dtype="int64", shape=("C", "T")),
            ),
            TensorFieldSpec(
                "candidate_scores",
                IOKind.OUTPUT,
                "candidate_scores",
                TensorContract(dtype="float32", shape=("C", "T")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=40,
            fields=(
                PushConstantFieldSpec("S", PushConstantType.UINT32, 0, "S"),
                PushConstantFieldSpec("C", PushConstantType.UINT32, 4, "C"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 8, "T"),
                PushConstantFieldSpec("V", PushConstantType.UINT32, 12, _AUDIO_VOCAB_SIZE),
                PushConstantFieldSpec("guidance_scale", PushConstantType.FLOAT32, 16, 2.0),
                PushConstantFieldSpec("layer_penalty", PushConstantType.FLOAT32, 20, 5.0),
                PushConstantFieldSpec("position_temperature", PushConstantType.FLOAT32, 24, 5.0),
                PushConstantFieldSpec(
                    "step_index", PushConstantType.UINT32, 28, PushConstantInput("step_index")
                ),
                PushConstantFieldSpec(
                    "rng_seed", PushConstantType.UINT32, 32, PushConstantInput("rng_seed")
                ),
                PushConstantFieldSpec(
                    "audio_mask_id",
                    PushConstantType.UINT32,
                    36,
                    PushConstantInput("audio_mask_id"),
                ),
            ),
        ),
        dispatch=(ceil_div(mul("C", "T"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        require_shader_int64=True,
        require_storage_buffer_16bit_access=True,
    ),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly LogitsBuffer {
    float16_t logits[];
};

layout(set = 0, binding = 1) buffer restrict readonly TokensBuffer {
    int64_t tokens[];
};

layout(set = 0, binding = 2) buffer restrict writeonly CandidateTokensBuffer {
    int64_t candidate_tokens[];
};

layout(set = 0, binding = 3) buffer restrict writeonly CandidateScoresBuffer {
    float candidate_scores[];
};

layout(push_constant) uniform PushConstants {
    uint S;
    uint C;
    uint T;
    uint V;
    float guidance_scale;
    float layer_penalty;
    float position_temperature;
    uint step_index;
    uint rng_seed;
    uint audio_mask_id;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

bool better_pair(float lhs_score, uint lhs_token, float rhs_score, uint rhs_token) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_token < rhs_token);
}

uint hash_u32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

float uniform01(uint x) {
    return (float(hash_u32(x) & 0x00ffffffu) + 0.5) * (1.0 / 16777216.0);
}

float gumbel_noise(uint flat_pos) {
    uint x = pc.rng_seed ^ (pc.step_index * 0x9e3779b9u) ^ (flat_pos * 0x85ebca6bu);
    const float u = clamp(uniform01(x), 1.0e-7, 1.0 - 1.0e-7);
    return -log(-log(u));
}

float guided_logit(uint batch, uint seq, uint codebook, uint token, float c_max, float c_sum, float u_max, float u_sum) {
    const uint offset = codebook * pc.V + token;
    const float c_logit = float(logits[(0u * pc.S + seq) * (pc.C * pc.V) + offset]);
    const float u_logit = float(logits[(1u * pc.S + batch) * (pc.C * pc.V) + offset]);
    const float c_log_prob = c_logit - c_max - log(c_sum);
    const float u_log_prob = u_logit - u_max - log(u_sum);
    return c_log_prob + pc.guidance_scale * (c_log_prob - u_log_prob);
}

void main() {
    const uint flat_pos = gl_GlobalInvocationID.x;
    const uint total = pc.C * pc.T;
    if (flat_pos >= total) {
        return;
    }

    const uint codebook = flat_pos / pc.T;
    const uint target = flat_pos - codebook * pc.T;
    const uint cond_seq = pc.S - pc.T + target;
    const uint uncond_seq = target;
    const uint vocab_offset = codebook * pc.V;
    const uint mask_token = pc.audio_mask_id;

    float c_max = -3.4028234663852886e+38;
    float u_max = -3.4028234663852886e+38;
    for (uint token = 0u; token < pc.V; ++token) {
        const uint offset = vocab_offset + token;
        c_max = max(c_max, float(logits[(0u * pc.S + cond_seq) * (pc.C * pc.V) + offset]));
        u_max = max(u_max, float(logits[(1u * pc.S + uncond_seq) * (pc.C * pc.V) + offset]));
    }

    float c_sum = 0.0;
    float u_sum = 0.0;
    for (uint token = 0u; token < pc.V; ++token) {
        const uint offset = vocab_offset + token;
        c_sum += exp(float(logits[(0u * pc.S + cond_seq) * (pc.C * pc.V) + offset]) - c_max);
        u_sum += exp(float(logits[(1u * pc.S + uncond_seq) * (pc.C * pc.V) + offset]) - u_max);
    }

    float guided_max = -3.4028234663852886e+38;
    for (uint token = 0u; token < pc.V; ++token) {
        guided_max = max(
            guided_max,
            guided_logit(uncond_seq, cond_seq, codebook, token, c_max, c_sum, u_max, u_sum)
        );
    }

    float guided_sum = 0.0;
    float best_score = -3.4028234663852886e+38;
    uint best_token = 0xffffffffu;
    for (uint token = 0u; token < pc.V; ++token) {
        const float score = guided_logit(uncond_seq, cond_seq, codebook, token, c_max, c_sum, u_max, u_sum);
        guided_sum += exp(score - guided_max);
        if (token != mask_token && better_pair(score, token, best_score, best_token)) {
            best_score = score;
            best_token = token;
        }
    }

    float confidence = best_score - guided_max - log(guided_sum);
    confidence -= float(codebook) * pc.layer_penalty;
    if (pc.position_temperature > 0.0) {
        confidence = confidence / pc.position_temperature + gumbel_noise(flat_pos);
    }
    if (tokens[flat_pos] != int64_t(pc.audio_mask_id)) {
        confidence = -3.4028234663852886e+38;
    }
    candidate_tokens[flat_pos] = int64_t(best_token);
    candidate_scores[flat_pos] = confidence;
}
""".lstrip(),
)


OMNIVOICE_TOKEN_UPDATE_TOPK_F32 = ShaderVariant(
    name="omnivoice_token_update_topk_f32",
    family="omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceTokenUpdateTopkF32Program",
        shader_name="omnivoice_token_update_topk_f32",
        fields=(
            TensorFieldSpec(
                "candidate_tokens",
                IOKind.INPUT,
                "candidate_tokens",
                TensorContract(dtype="int64", shape=("C", "T")),
            ),
            TensorFieldSpec(
                "candidate_scores",
                IOKind.INPUT,
                "candidate_scores",
                TensorContract(dtype="float32", shape=("C", "T")),
            ),
            TensorFieldSpec(
                "tokens",
                IOKind.INOUT,
                "tokens",
                TensorContract(dtype="int64", shape=(1, "C", "T")),
            ),
            TensorFieldSpec(
                "batch_input_ids",
                IOKind.INOUT,
                "tokens",
                TensorContract(dtype="int64", shape=(2, "C", "S")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec("C", PushConstantType.UINT32, 0, "C"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 8, "S"),
                PushConstantFieldSpec(
                    "unmask_count",
                    PushConstantType.UINT32,
                    12,
                    PushConstantInput("unmask_count"),
                ),
            ),
        ),
        dispatch=(ceil_div(mul("C", "T"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly CandidateTokensBuffer {
    int64_t candidate_tokens[];
};

layout(set = 0, binding = 1) buffer restrict readonly CandidateScoresBuffer {
    float candidate_scores[];
};

layout(set = 0, binding = 2) buffer restrict TokensBuffer {
    int64_t tokens[];
};

layout(set = 0, binding = 3) buffer restrict BatchInputIdsBuffer {
    int64_t batch_input_ids[];
};

layout(push_constant) uniform PushConstants {
    uint C;
    uint T;
    uint S;
    uint unmask_count;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

bool better_pair(float lhs_score, uint lhs_index, float rhs_score, uint rhs_index) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_index < rhs_index);
}

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint total = pc.C * pc.T;
    const uint limit = min(pc.unmask_count, total);
    if (index >= total) {
        return;
    }

    const float score = candidate_scores[index];
    uint rank = 0u;
    for (uint other = 0u; other < total; ++other) {
        if (better_pair(candidate_scores[other], other, score, index)) {
            ++rank;
        }
    }

    if (rank < limit) {
        const uint codebook = index / pc.T;
        const uint target = index - codebook * pc.T;
        const int64_t token = candidate_tokens[index];
        tokens[index] = token;
        batch_input_ids[(0u * pc.C + codebook) * pc.S + (pc.S - pc.T + target)] = token;
        batch_input_ids[(1u * pc.C + codebook) * pc.S + target] = token;
    }
}
""".lstrip(),
)
