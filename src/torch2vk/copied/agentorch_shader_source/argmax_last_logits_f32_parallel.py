"""Parallel two-pass argmax for the last row of F32 logits."""

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


ARGMAX_LAST_LOGITS_F32_CHUNK = 1024


ARGMAX_LAST_LOGITS_F32_STAGE1 = shader_variant(
    name="argmax_last_logits_f32_stage1",
    family="argmax_last_logits_f32_parallel",
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=32, require_full_subgroups=True)
    ),
    contract=shader_contract(
        class_name="ArgmaxLastLogitsF32Stage1Program",
        shader_name="argmax_last_logits_f32_stage1",
        fields=(
            input_tensor(
                name="logits",
                binding="t_logits",
                role="logits",
                dtypes=("float32",),
                shape=("B", "S", "V"),
            ),
            output_tensor(
                name="partial_values",
                binding="t_partial_values",
                role="partial_values",
                dtypes=("float32",),
                shape=("B", "C"),
            ),
            output_tensor(
                name="partial_indices",
                binding="t_partial_indices",
                role="partial_indices",
                dtypes=("int32",),
                shape=("B", "C"),
            ),
        ),
        uniforms=(uniform_ivec4(binding="sizes", value=("V", "S", "C", 1)),),
        dispatch=("C", "B", 1),
        bindings=(
            storage_buffer_binding(name="t_logits", binding=0),
            storage_buffer_binding(name="t_partial_values", binding=1),
            storage_buffer_binding(name="t_partial_indices", binding=2),
            uniform_buffer_binding(name="sizes", binding=3),
        ),
    ),
    source=f"""
#version 460

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly LogitsBuffer {{
    float t_logits[];
}};

layout(set = 0, binding = 1) buffer restrict writeonly PartialValuesBuffer {{
    float t_partial_values[];
}};

layout(set = 0, binding = 2) buffer restrict writeonly PartialIndicesBuffer {{
    int t_partial_indices[];
}};

layout(set = 0, binding = 3) uniform restrict readonly sizes_UBO {{
    ivec4 sizes;
}};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const uint CHUNK_SIZE = {ARGMAX_LAST_LOGITS_F32_CHUNK}u;
const uint SUBGROUPS_PER_WORKGROUP = 8u;
const float NEG_INF = -3.402823466e+38;

shared float shared_best_value[SUBGROUPS_PER_WORKGROUP];
shared int shared_best_index[SUBGROUPS_PER_WORKGROUP];

bool candidate_beats(
    const float candidate_value,
    const int candidate_index,
    const float current_value,
    const int current_index
) {{
    return candidate_value > current_value
        || (candidate_value == current_value && candidate_index < current_index);
}}

void main() {{
    const uint lane = gl_LocalInvocationID.x;
    const uint subgroup_lane = gl_SubgroupInvocationID;
    const uint subgroup_id = gl_SubgroupID;
    const uint chunk_index = gl_WorkGroupID.x;
    const uint batch_index = gl_WorkGroupID.y;

    const uint vocab = uint(sizes.x);
    const uint steps = uint(sizes.y);
    const uint chunks = uint(sizes.z);

    if (steps == 0u || vocab == 0u || chunk_index >= chunks) {{
        return;
    }}

    const uint start = chunk_index * CHUNK_SIZE;
    const uint end = min(start + CHUNK_SIZE, vocab);
    const uint logits_base = ((batch_index * steps) + (steps - 1u)) * vocab;

    float best_value = NEG_INF;
    int best_index = 0;
    for (uint vocab_index = start + lane; vocab_index < end; vocab_index += 256u) {{
        const float value = t_logits[logits_base + vocab_index];
        if (candidate_beats(value, int(vocab_index), best_value, best_index)) {{
            best_value = value;
            best_index = int(vocab_index);
        }}
    }}

    const float subgroup_best_value = subgroupMax(best_value);
    int subgroup_best_index = best_index;
    if (best_value != subgroup_best_value) {{
        subgroup_best_index = 2147483647;
    }}
    subgroup_best_index = subgroupMin(subgroup_best_index);

    if (subgroup_lane == 0u) {{
        shared_best_value[subgroup_id] = subgroup_best_value;
        shared_best_index[subgroup_id] = subgroup_best_index;
    }}
    barrier();

    if (lane == 0u) {{
        float final_best_value = shared_best_value[0];
        int final_best_index = shared_best_index[0];
        for (uint subgroup = 1u; subgroup < SUBGROUPS_PER_WORKGROUP; ++subgroup) {{
            const float other_value = shared_best_value[subgroup];
            const int other_index = shared_best_index[subgroup];
            if (candidate_beats(other_value, other_index, final_best_value, final_best_index)) {{
                final_best_value = other_value;
                final_best_index = other_index;
            }}
        }}
        const uint partial_offset = batch_index * chunks + chunk_index;
        t_partial_values[partial_offset] = final_best_value;
        t_partial_indices[partial_offset] = final_best_index;
    }}
}}
""".lstrip(),
)


ARGMAX_LAST_LOGITS_F32_STAGE1_FORBID_TOKENS = shader_variant(
    name="argmax_last_logits_f32_stage1_forbid_tokens",
    family="argmax_last_logits_f32_parallel",
    execution_requirements=ARGMAX_LAST_LOGITS_F32_STAGE1.execution_requirements,
    contract=shader_contract(
        class_name="ArgmaxLastLogitsF32Stage1ForbidTokensProgram",
        shader_name="argmax_last_logits_f32_stage1_forbid_tokens",
        fields=(
            input_tensor(name="logits", binding="t_logits", role="logits", dtypes=("float32",), shape=("B", "S", "V")),
            output_tensor(name="partial_values", binding="t_partial_values", role="partial_values", dtypes=("float32",), shape=("B", "C")),
            output_tensor(name="partial_indices", binding="t_partial_indices", role="partial_indices", dtypes=("int32",), shape=("B", "C")),
            input_tensor(name="forbidden_token_ids", binding="t_forbidden", role="forbidden_token_ids", dtypes=("int32",), shape=("F",)),
        ),
        uniforms=(uniform_ivec4(binding="sizes", value=("V", "S", "C", "F")),),
        dispatch=("C", "B", 1),
        bindings=(
            storage_buffer_binding(name="t_logits", binding=0),
            storage_buffer_binding(name="t_partial_values", binding=1),
            storage_buffer_binding(name="t_partial_indices", binding=2),
            storage_buffer_binding(name="t_forbidden", binding=3),
            uniform_buffer_binding(name="sizes", binding=4),
        ),
    ),
    source=ARGMAX_LAST_LOGITS_F32_STAGE1.source.replace(
        "layout(set = 0, binding = 3) uniform restrict readonly sizes_UBO {",
        """layout(set = 0, binding = 3) buffer restrict readonly ForbiddenTokenBuffer {
    int t_forbidden[];
};

layout(set = 0, binding = 4) uniform restrict readonly sizes_UBO {""",
    )
    .replace(
        "const uint chunks = uint(sizes.z);",
        "const uint chunks = uint(sizes.z);\n    const uint forbidden_count = uint(sizes.w);",
    )
    .replace(
        "const float value = t_logits[logits_base + vocab_index];",
        """bool is_forbidden = false;
        for (uint forbidden_index = 0u; forbidden_index < forbidden_count; ++forbidden_index) {
            is_forbidden = is_forbidden || int(vocab_index) == t_forbidden[forbidden_index];
        }
        const float value = is_forbidden ? NEG_INF : t_logits[logits_base + vocab_index];""",
    ),
)


ARGMAX_LAST_LOGITS_F32_STAGE2 = shader_variant(
    name="argmax_last_logits_f32_stage2",
    family="argmax_last_logits_f32_parallel",
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=32, require_full_subgroups=True)
    ),
    contract=shader_contract(
        class_name="ArgmaxLastLogitsF32Stage2Program",
        shader_name="argmax_last_logits_f32_stage2",
        fields=(
            input_tensor(
                name="partial_values",
                binding="t_partial_values",
                role="partial_values",
                dtypes=("float32",),
                shape=("B", "C"),
            ),
            input_tensor(
                name="partial_indices",
                binding="t_partial_indices",
                role="partial_indices",
                dtypes=("int32",),
                shape=("B", "C"),
            ),
            output_tensor(
                name="output",
                binding="t_output",
                role="token_id",
                dtypes=("int32",),
                shape=("B",),
            ),
        ),
        uniforms=(uniform_ivec4(binding="sizes", value=("C", "B", 1, 1)),),
        dispatch=(1, 1, "B"),
        bindings=(
            storage_buffer_binding(name="t_partial_values", binding=0),
            storage_buffer_binding(name="t_partial_indices", binding=1),
            storage_buffer_binding(name="t_output", binding=2),
            uniform_buffer_binding(name="sizes", binding=3),
        ),
    ),
    source="""
#version 460

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly PartialValuesBuffer {
    float t_partial_values[];
};

layout(set = 0, binding = 1) buffer restrict readonly PartialIndicesBuffer {
    int t_partial_indices[];
};

layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer {
    int t_output[];
};

layout(set = 0, binding = 3) uniform restrict readonly sizes_UBO {
    ivec4 sizes;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const uint SUBGROUPS_PER_WORKGROUP = 8u;
const float NEG_INF = -3.402823466e+38;

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

    const uint chunks = uint(sizes.x);
    const uint batches = uint(sizes.y);

    if (batch_index >= batches || chunks == 0u) {
        return;
    }

    float best_value = NEG_INF;
    int best_index = 0;
    const uint partial_base = batch_index * chunks;
    for (uint chunk_index = lane; chunk_index < chunks; chunk_index += 256u) {
        const float value = t_partial_values[partial_base + chunk_index];
        const int index = t_partial_indices[partial_base + chunk_index];
        if (candidate_beats(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
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
