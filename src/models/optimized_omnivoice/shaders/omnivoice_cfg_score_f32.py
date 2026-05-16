"""Generated shader: omnivoice_cfg_score_f32."""

from __future__ import annotations

from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantInput,
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
    ShaderExecutionRequirements,
)


OMNIVOICE_CFG_SCORE_F32 = ShaderVariant(
    name="omnivoice_cfg_score_f32",
    family="omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceCfgScoreF32Program",
        shader_name="omnivoice_cfg_score_f32",
        fields=(
            TensorFieldSpec(
                name="logits",
                io_kind=IOKind.INPUT,
                role="logits",
                contract=TensorContract(
                    dtype="float16",
                    shape=(
                        2,
                        "S",
                        "CV",
                    ),
                ),
            ),
            TensorFieldSpec(
                name="tokens",
                io_kind=IOKind.INPUT,
                role="tokens",
                contract=TensorContract(
                    dtype="int64",
                    shape=(
                        1,
                        "C",
                        "T",
                    ),
                ),
            ),
            TensorFieldSpec(
                name="audio_mask_id",
                io_kind=IOKind.INPUT,
                role="mask_id",
                contract=TensorContract(dtype="int64", shape=(1,)),
            ),
            TensorFieldSpec(
                name="active_target_len",
                io_kind=IOKind.INPUT,
                role="active_target_len",
                contract=TensorContract(dtype="uint32", shape=(1,)),
            ),
            TensorFieldSpec(
                name="cond_target_start",
                io_kind=IOKind.INPUT,
                role="cond_target_start",
                contract=TensorContract(dtype="uint32", shape=(1,)),
            ),
            TensorFieldSpec(
                name="candidate_tokens",
                io_kind=IOKind.OUTPUT,
                role="candidate_tokens",
                contract=TensorContract(
                    dtype="int64",
                    shape=(
                        "C",
                        "T",
                    ),
                ),
            ),
            TensorFieldSpec(
                name="candidate_scores",
                io_kind=IOKind.OUTPUT,
                role="candidate_scores",
                contract=TensorContract(
                    dtype="float32",
                    shape=(
                        "C",
                        "T",
                    ),
                ),
            ),
        ),
        push_constants=PushConstantSpec(
            size=36,
            fields=(
                PushConstantFieldSpec("S", PushConstantType.UINT32, 0, "S", dynamic=False),
                PushConstantFieldSpec("C", PushConstantType.UINT32, 4, "C", dynamic=False),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 8, "T", dynamic=False),
                PushConstantFieldSpec("V", PushConstantType.UINT32, 12, 1025, dynamic=False),
                PushConstantFieldSpec(
                    "guidance_scale", PushConstantType.FLOAT32, 16, 2.0, dynamic=False
                ),
                PushConstantFieldSpec(
                    "layer_penalty", PushConstantType.FLOAT32, 20, 5.0, dynamic=False
                ),
                PushConstantFieldSpec(
                    "position_temperature", PushConstantType.FLOAT32, 24, 5.0, dynamic=False
                ),
                PushConstantFieldSpec(
                    "step_index",
                    PushConstantType.UINT32,
                    28,
                    PushConstantInput("step_index"),
                    dynamic=False,
                ),
                PushConstantFieldSpec(
                    "rng_seed",
                    PushConstantType.UINT32,
                    32,
                    PushConstantInput("rng_seed"),
                    dynamic=False,
                ),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul("C", "T"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        require_shader_int64=True, require_storage_buffer_16bit_access=True
    ),
    source="""\
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

layout(set = 0, binding = 2) buffer restrict readonly AudioMaskIdBuffer {
    int64_t audio_mask_id[];
};

layout(set = 0, binding = 3) buffer restrict readonly ActiveTargetLenBuffer {
    uint active_target_len[];
};

layout(set = 0, binding = 4) buffer restrict readonly CondTargetStartBuffer {
    uint cond_target_start[];
};

layout(set = 0, binding = 5) buffer restrict writeonly CandidateTokensBuffer {
    int64_t candidate_tokens[];
};

layout(set = 0, binding = 6) buffer restrict writeonly CandidateScoresBuffer {
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
    const uint mask_token = uint(audio_mask_id[0]);
    if (target >= active_target_len[0]) {
        candidate_tokens[flat_pos] = int64_t(mask_token);
        candidate_scores[flat_pos] = -3.4028234663852886e+38;
        return;
    }
    const uint cond_seq = cond_target_start[0] + target;
    const uint uncond_seq = target;
    const uint vocab_offset = codebook * pc.V;

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
        const uint active_flat_pos = codebook * active_target_len[0] + target;
        confidence = confidence / pc.position_temperature + gumbel_noise(active_flat_pos);
    }
    if (tokens[flat_pos] != audio_mask_id[0]) {
        confidence = -3.4028234663852886e+38;
    }
    candidate_tokens[flat_pos] = int64_t(best_token);
    candidate_scores[flat_pos] = confidence;
}
""",
)
