"""Qwen3 argmax stage 2 shader."""

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
"""

ARGMAX_LAST_LOGITS_STAGE2 = ShaderVariant(
    name="argmax_last_logits_f32_stage2",
    family="argmax",
    contract=ShaderContract(
        name="argmax_last_logits_f32_stage2",
        inputs={
            "partial_values": TensorContract(dtype="float32", shape=("B", "C")),
            "partial_indices": TensorContract(dtype="int32", shape=("B", "C")),
        },
        outputs={"output": TensorContract(dtype="int32", shape=("B",))},
        bindings=(
            Binding("partial_values", 0, BindingAccess.READ),
            Binding("partial_indices", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=(1, 1, "B"),
        uniforms=(UniformBlock("sizes", 3, ("C", "B", 1, 1)),),
    ),
    source=_SOURCE,
)
