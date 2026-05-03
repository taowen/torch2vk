"""Embedded GLSL for argmax_last_logits_f32."""

from __future__ import annotations

from agentorch.kernel.contract import (
    input_tensor,
    output_tensor,
    shader_contract,
    storage_buffer_binding,
    uniform_buffer_binding,
    uniform_ivec4,
)

from .shader_execution_requirements import ShaderExecutionRequirements, SubgroupRequirements
from .shader_variant import shader_variant


ARGMAX_LAST_LOGITS_F32 = shader_variant(
    name="argmax_last_logits_f32",
    family="argmax_last_logits_f32",
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=32, require_full_subgroups=True)
    ),
    contract=shader_contract(
        class_name="ArgmaxLastLogitsF32Program",
        shader_name="argmax_last_logits_f32",
        fields=(
            input_tensor(
                name="logits",
                binding="t_logits",
                role="logits",
                dtypes=("float32",),
                shape=("B", "S", "V"),
            ),
            output_tensor(
                name="output",
                binding="t_output",
                role="token_id",
                dtypes=("int32",),
                shape=("B",),
            ),
        ),
        uniforms=(uniform_ivec4(binding="sizes", value=("V", "S", "B", 1)),),
        dispatch=(1, 1, "B"),
        bindings=(
            storage_buffer_binding(name="t_output", binding=0),
            storage_buffer_binding(name="t_logits", binding=1),
            uniform_buffer_binding(name="sizes", binding=2),
        ),
    ),
    source="""
#version 460

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer {
    int t_output[];
};

layout(set = 0, binding = 1) buffer restrict readonly LogitsBuffer {
    float t_logits[];
};

layout(set = 0, binding = 2) uniform restrict readonly sizes_UBO {
    ivec4 sizes;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const uint SUBGROUPS_PER_WORKGROUP = 8u;

shared float shared_best_value[SUBGROUPS_PER_WORKGROUP];
shared int shared_best_index[SUBGROUPS_PER_WORKGROUP];

bool candidate_beats(
    const float candidate_value,
    const int candidate_index,
    const float current_value,
    const int current_index
) {
    return candidate_value > current_value
        || (candidate_value == current_value && candidate_index < current_index);
}

void main() {
    const uint lane = gl_LocalInvocationID.x;
    const uint subgroup_lane = gl_SubgroupInvocationID;
    const uint subgroup_id = gl_SubgroupID;
    const uint batch_index = gl_WorkGroupID.z;

    const uint vocab = uint(sizes.x);
    const uint steps = uint(sizes.y);
    const uint batches = uint(sizes.z);

    if (batch_index >= batches || steps == 0u || vocab == 0u) {
        return;
    }

    const uint last_step = steps - 1u;
    const uint logits_base = ((batch_index * steps) + last_step) * vocab;

    float best_value = -3.402823466e+38;
    int best_index = 0;
    for (uint vocab_index = lane; vocab_index < vocab; vocab_index += 256u) {
        const float value = t_logits[logits_base + vocab_index];
        if (candidate_beats(value, int(vocab_index), best_value, best_index)) {
            best_value = value;
            best_index = int(vocab_index);
        }
    }

    const float subgroup_best_value = subgroupMax(best_value);
    int subgroup_best_index = best_index;
    if (best_value != subgroup_best_value) {
        subgroup_best_index = 2147483647;
    }
    subgroup_best_index = subgroupMin(subgroup_best_index);

    if (subgroup_lane == 0u) {
        shared_best_value[subgroup_id] = subgroup_best_value;
        shared_best_index[subgroup_id] = subgroup_best_index;
    }
    barrier();

    if (lane == 0u) {
        float final_best_value = shared_best_value[0];
        int final_best_index = shared_best_index[0];
        for (uint subgroup = 1u; subgroup < SUBGROUPS_PER_WORKGROUP; ++subgroup) {
            const float other_value = shared_best_value[subgroup];
            const int other_index = shared_best_index[subgroup];
            if (candidate_beats(other_value, other_index, final_best_value, final_best_index)) {
                final_best_value = other_value;
                final_best_index = other_index;
            }
        }
        t_output[batch_index] = final_best_index;
    }
}
""".lstrip(),
)


ARGMAX_LAST_LOGITS_F32_FORBID_TOKENS = shader_variant(
    name="argmax_last_logits_f32_forbid_tokens",
    family="argmax_last_logits_f32",
    execution_requirements=ARGMAX_LAST_LOGITS_F32.execution_requirements,
    contract=shader_contract(
        class_name="ArgmaxLastLogitsF32ForbidTokensProgram",
        shader_name="argmax_last_logits_f32_forbid_tokens",
        fields=(
            input_tensor(name="logits", binding="t_logits", role="logits", dtypes=("float32",), shape=("B", "S", "V")),
            input_tensor(name="forbidden_token_ids", binding="t_forbidden", role="forbidden_token_ids", dtypes=("int32",), shape=("F",)),
            output_tensor(name="output", binding="t_output", role="token_id", dtypes=("int32",), shape=("B",)),
        ),
        uniforms=(uniform_ivec4(binding="sizes", value=("V", "S", "B", "F")),),
        dispatch=(1, 1, "B"),
        bindings=(
            storage_buffer_binding(name="t_output", binding=0),
            storage_buffer_binding(name="t_logits", binding=1),
            storage_buffer_binding(name="t_forbidden", binding=2),
            uniform_buffer_binding(name="sizes", binding=3),
        ),
    ),
    source=ARGMAX_LAST_LOGITS_F32.source.replace(
        "layout(set = 0, binding = 2) uniform restrict readonly sizes_UBO {",
        """layout(set = 0, binding = 2) buffer restrict readonly ForbiddenTokenBuffer {
    int t_forbidden[];
};

layout(set = 0, binding = 3) uniform restrict readonly sizes_UBO {""",
    )
    .replace(
        "const uint batches = uint(sizes.z);",
        "const uint batches = uint(sizes.z);\n    const uint forbidden_count = uint(sizes.w);",
    )
    .replace(
        "const float value = t_logits[logits_base + vocab_index];",
        """bool is_forbidden = false;
        for (uint forbidden_index = 0u; forbidden_index < forbidden_count; ++forbidden_index) {
            is_forbidden = is_forbidden || int(vocab_index) == t_forbidden[forbidden_index];
        }
        const float value = is_forbidden ? -3.402823466e+38 : t_logits[logits_base + vocab_index];""",
    ),
)
