"""Generated shader: omnivoice_token_update_topk_f32."""

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
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


OMNIVOICE_TOKEN_UPDATE_TOPK_F32 = ShaderVariant(
    name='omnivoice_token_update_topk_f32',
    family='omnivoice',
    contract=ShaderContract(
        class_name='OmniVoiceTokenUpdateTopkF32Program',
        shader_name='omnivoice_token_update_topk_f32',
        fields=(
            TensorFieldSpec(
                name='candidate_tokens',
                io_kind=IOKind.INPUT,
                role='candidate_tokens',
                contract=TensorContract(dtype='int64', shape=('C', 'T',)),
            ),
            TensorFieldSpec(
                name='candidate_scores',
                io_kind=IOKind.INPUT,
                role='candidate_scores',
                contract=TensorContract(dtype='float32', shape=('C', 'T',)),
            ),
            TensorFieldSpec(
                name='unmask_count',
                io_kind=IOKind.INPUT,
                role='unmask_count',
                contract=TensorContract(dtype='uint32', shape=(1,)),
            ),
            TensorFieldSpec(
                name='tokens',
                io_kind=IOKind.INOUT,
                role='tokens',
                contract=TensorContract(dtype='int64', shape=(1, 'C', 'T',)),
            ),
            TensorFieldSpec(
                name='batch_input_ids',
                io_kind=IOKind.INOUT,
                role='tokens',
                contract=TensorContract(dtype='int64', shape=(2, 'C', 'S',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec('C', PushConstantType.UINT32, 0, 'C', dynamic=False),
                PushConstantFieldSpec('T', PushConstantType.UINT32, 4, 'T', dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 8, 'S', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul('C', 'T'), 256), 1, 1),
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

layout(set = 0, binding = 2) buffer restrict readonly UnmaskCountBuffer {
    uint unmask_count[];
};

layout(set = 0, binding = 3) buffer restrict TokensBuffer {
    int64_t tokens[];
};

layout(set = 0, binding = 4) buffer restrict BatchInputIdsBuffer {
    int64_t batch_input_ids[];
};

layout(push_constant) uniform PushConstants {
    uint C;
    uint T;
    uint S;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

bool better_pair(float lhs_score, uint lhs_index, float rhs_score, uint rhs_index) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_index < rhs_index);
}

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint total = pc.C * pc.T;
    const uint limit = min(unmask_count[0], total);
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
""",
)
