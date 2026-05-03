"""Omnivoice Argmax Select Apply Fused L."""

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

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly OutputUpdatedIdsBuffer { int t_output_updated_ids[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputSelectedFlatIndexBuffer { int t_output_selected_flat_index[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputSelectedScoreBuffer { float t_output_selected_score[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputSelectedCandidateIdBuffer { int t_output_selected_candidate_id[]; };
layout(set = 0, binding = 4) buffer restrict readonly LogitsBuffer { float t_logits[]; };
layout(set = 0, binding = 5) buffer restrict readonly CodebookOffsetsBuffer { int t_codebook_offsets[]; };
layout(set = 0, binding = 6) buffer restrict readonly PenaltyBuffer { float t_penalty[]; };
layout(set = 0, binding = 7) buffer restrict readonly CurrentIdsBuffer { int t_current_ids[]; };
layout(set = 0, binding = 8) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared float s_reduce_best_score[256];
shared int s_reduce_best_cs[256];
shared int s_selected_cs;
shared int s_selected_id;
shared float s_selected_penalized_score;

void main() {
    const uint batch = gl_WorkGroupID.z;
    const uint lane = gl_LocalInvocationID.x;
    const uint steps = uint(sizes.x);
    const uint codebooks = uint(sizes.y);
    const uint batches = uint(sizes.z);
    const uint vocab = uint(sizes.w);
    if (batch >= batches) {
        return;
    }

    const uint cs_len = codebooks * steps;
    const uint logits_batch_base = batch * steps * vocab;

    float reduce_best = -3.402823466e38;
    int reduce_cs = 0;
    for (uint cs = lane; cs < cs_len; cs += 256u) {
        const uint codebook = cs / steps;
        const uint step = cs - codebook * steps;
        const int start = t_codebook_offsets[codebook];
        const int end = (codebook + 1u < codebooks) ? t_codebook_offsets[codebook + 1u] : int(vocab);
        const int seg_len = max(0, end - start);
        const uint logits_base = logits_batch_base + step * vocab;
        float best_val = -3.402823466e38;
        for (int i = 0; i < seg_len; ++i) {
            const float v = t_logits[logits_base + uint(start + i)];
            if (v > best_val) {
                best_val = v;
            }
        }
        const float penalized = best_val - t_penalty[codebook];
        if (penalized > reduce_best) {
            reduce_best = penalized;
            reduce_cs = int(cs);
        }
    }
    s_reduce_best_score[lane] = reduce_best;
    s_reduce_best_cs[lane] = reduce_cs;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            const float other = s_reduce_best_score[lane + stride];
            if (other > s_reduce_best_score[lane]) {
                s_reduce_best_score[lane] = other;
                s_reduce_best_cs[lane] = s_reduce_best_cs[lane + stride];
            }
        }
        barrier();
    }

    if (lane == 0u) {
        const int selected_cs = s_reduce_best_cs[0];
        const uint selected_codebook = uint(selected_cs) / steps;
        const uint selected_step = uint(selected_cs) - selected_codebook * steps;
        const int start = t_codebook_offsets[selected_codebook];
        const int end = (selected_codebook + 1u < codebooks) ? t_codebook_offsets[selected_codebook + 1u] : int(vocab);
        const int seg_len = max(0, end - start);
        const uint logits_base = logits_batch_base + selected_step * vocab;
        float best_val = -3.402823466e38;
        int best_id = 0;
        for (int i = 0; i < seg_len; ++i) {
            const float v = t_logits[logits_base + uint(start + i)];
            if (v > best_val) {
                best_val = v;
                best_id = i;
            }
        }
        s_selected_cs = selected_cs;
        s_selected_id = best_id;
        s_selected_penalized_score = s_reduce_best_score[0];
        t_output_selected_flat_index[batch] = selected_cs;
        t_output_selected_score[batch] = s_selected_penalized_score;
        t_output_selected_candidate_id[batch] = best_id;
    }
    barrier();

    for (uint cs = lane; cs < cs_len; cs += 256u) {
        const uint codebook = cs / steps;
        const uint step = cs - codebook * steps;
        const uint out_idx = (batch * codebooks + codebook) * steps + step;
        t_output_updated_ids[out_idx] = (int(cs) == s_selected_cs) ? s_selected_id : t_current_ids[out_idx];
    }
}
"""


OMNIVOICE_ARGMAX_SELECT_APPLY_FUSED_L = ShaderVariant(
    name="omnivoice_argmax_select_apply_fused_l",
    family="omnivoice_iterative_update",
    contract=ShaderContract(
        name="omnivoice_argmax_select_apply_fused_l",
        inputs={
            "logits": TensorContract(dtype="float32", shape=("B", "S", "V")),
            "codebook_offsets": TensorContract(dtype="int32", shape=("C",)),
            "penalty": TensorContract(dtype="float32", shape=("C",)),
            "current_ids": TensorContract(dtype="int32", shape=("B", "C", "S")),
        },
        outputs={
            "output_updated_ids": TensorContract(dtype="int32", shape=("B", "C", "S")),
            "output_selected_flat_index": TensorContract(dtype="int32", shape=("B",)),
            "output_selected_score": TensorContract(dtype="float32", shape=("B",)),
            "output_selected_candidate_id": TensorContract(dtype="int32", shape=("B",)),
        },
        bindings=(
            Binding("output_updated_ids", 0, BindingAccess.WRITE),
            Binding("output_selected_flat_index", 1, BindingAccess.WRITE),
            Binding("output_selected_score", 2, BindingAccess.WRITE),
            Binding("output_selected_candidate_id", 3, BindingAccess.WRITE),
            Binding("logits", 4, BindingAccess.READ),
            Binding("codebook_offsets", 5, BindingAccess.READ),
            Binding("penalty", 6, BindingAccess.READ),
            Binding("current_ids", 7, BindingAccess.READ),
        ),
        uniforms=(UniformBlock("sizes", 8, ("S", "C", "B", "V")),),
        dispatch=(1, 1, "B"),
        push_constants=None,
    ),
    source=_SOURCE,
)
