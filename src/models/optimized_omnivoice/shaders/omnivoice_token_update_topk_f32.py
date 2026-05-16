"""Generated shader: omnivoice_token_update_topk_f32."""

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


OMNIVOICE_TOKEN_UPDATE_TOPK_F32 = ShaderVariant(
    name="omnivoice_token_update_topk_f32",
    family="omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceTokenUpdateTopkF32Program",
        shader_name="omnivoice_token_update_topk_f32",
        fields=(
            TensorFieldSpec(
                name="candidate_tokens",
                io_kind=IOKind.INPUT,
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
                io_kind=IOKind.INPUT,
                role="candidate_scores",
                contract=TensorContract(
                    dtype="float32",
                    shape=(
                        "C",
                        "T",
                    ),
                ),
            ),
            TensorFieldSpec(
                name="tokens",
                io_kind=IOKind.INOUT,
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
                name="batch_input_ids",
                io_kind=IOKind.INOUT,
                role="tokens",
                contract=TensorContract(
                    dtype="int64",
                    shape=(
                        2,
                        "C",
                        "S",
                    ),
                ),
            ),
        ),
        push_constants=PushConstantSpec(
            size=24,
            fields=(
                PushConstantFieldSpec("C", PushConstantType.UINT32, 0, "C", dynamic=False),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T", dynamic=False),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 8, "S", dynamic=False),
                PushConstantFieldSpec(
                    "unmask_count",
                    PushConstantType.UINT32,
                    12,
                    PushConstantInput("unmask_count"),
                    dynamic=False,
                ),
                PushConstantFieldSpec(
                    "active_target_len",
                    PushConstantType.UINT32,
                    16,
                    PushConstantInput("active_target_len"),
                    dynamic=False,
                ),
                PushConstantFieldSpec(
                    "cond_target_start",
                    PushConstantType.UINT32,
                    20,
                    PushConstantInput("cond_target_start"),
                    dynamic=False,
                ),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul("C", "T"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""\
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
    uint active_target_len;
    uint cond_target_start;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

bool better_pair(float lhs_score, uint lhs_index, float rhs_score, uint rhs_index) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_index < rhs_index);
}

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint active_t = pc.active_target_len;
    const uint total = pc.C * active_t;
    const uint limit = min(pc.unmask_count, total);
    if (index >= pc.C * pc.T) {
        return;
    }
    const uint codebook = index / pc.T;
    const uint target = index - codebook * pc.T;
    if (target >= active_t) {
        return;
    }

    const float score = candidate_scores[index];
    const uint active_index = codebook * active_t + target;
    uint rank = 0u;
    for (uint other = 0u; other < total; ++other) {
        const uint other_codebook = other / active_t;
        const uint other_target = other - other_codebook * active_t;
        const uint other_index = other_codebook * pc.T + other_target;
        if (better_pair(candidate_scores[other_index], other, score, active_index)) {
            ++rank;
        }
    }

    if (rank < limit) {
        const int64_t token = candidate_tokens[index];
        tokens[index] = token;
        batch_input_ids[(0u * pc.C + codebook) * pc.S + (pc.cond_target_start + target)] = token;
        batch_input_ids[(1u * pc.C + codebook) * pc.S + target] = token;
    }
}
""",
)
