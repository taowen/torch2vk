"""Qwen3 argmax stage 1 shader."""

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

layout(set = 0, binding = 0) buffer restrict readonly LogitsBuffer {
    float t_logits[];
};

layout(set = 0, binding = 1) buffer restrict writeonly PartialValuesBuffer {
    float t_partial_values[];
};

layout(set = 0, binding = 2) buffer restrict writeonly PartialIndicesBuffer {
    int t_partial_indices[];
};

layout(set = 0, binding = 3) uniform restrict readonly sizes_UBO {
    ivec4 sizes;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const uint CHUNK_SIZE = 1024u;
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
    const uint chunk_index = gl_WorkGroupID.x;
    const uint batch_index = gl_WorkGroupID.y;

    const uint vocab = uint(sizes.x);
    const uint steps = uint(sizes.y);
    const uint chunks = uint(sizes.z);

    if (steps == 0u || vocab == 0u || chunk_index >= chunks) {
        return;
    }

    const uint start = chunk_index * CHUNK_SIZE;
    const uint end = min(start + CHUNK_SIZE, vocab);
    const uint logits_base = ((batch_index * steps) + (steps - 1u)) * vocab;

    float best_value = NEG_INF;
    int best_index = 0;
    for (uint vocab_index = start + lane; vocab_index < end; vocab_index += 256u) {
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
        const uint partial_offset = batch_index * chunks + chunk_index;
        t_partial_values[partial_offset] = final_best_value;
        t_partial_indices[partial_offset] = final_best_index;
    }
}
"""

ARGMAX_LAST_LOGITS_STAGE1 = ShaderVariant(
    name="argmax_last_logits_f32_stage1",
    family="argmax",
    contract=ShaderContract(
        name="argmax_last_logits_f32_stage1",
        inputs={"logits": TensorContract(dtype="float32", shape=("B", "S", "V"))},
        outputs={
            "partial_values": TensorContract(dtype="float32", shape=("B", "C")),
            "partial_indices": TensorContract(dtype="int32", shape=("B", "C")),
        },
        bindings=(
            Binding("logits", 0, BindingAccess.READ),
            Binding("partial_values", 1, BindingAccess.WRITE),
            Binding("partial_indices", 2, BindingAccess.WRITE),
        ),
        dispatch=("C", "B", 1),
        uniforms=(UniformBlock("sizes", 3, ("V", "S", "C", 1)),),
    ),
    source=_SOURCE,
)
