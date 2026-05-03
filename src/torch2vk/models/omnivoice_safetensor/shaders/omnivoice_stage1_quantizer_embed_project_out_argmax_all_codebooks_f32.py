"""Omnivoice Stage1 Quantizer Embed Project Out Argmax All Codebooks F32."""

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

layout(set = 0, binding = 0) buffer restrict writeonly OutputIdsBuffer { int t_output_ids[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputScoresBuffer { float t_output_scores[]; };
layout(set = 0, binding = 2) buffer restrict readonly AudioIdsBuffer { int t_audio_ids[]; };
layout(set = 0, binding = 3) buffer restrict readonly Embed0Buffer { float t_embed0[]; };
layout(set = 0, binding = 4) buffer restrict readonly Embed1Buffer { float t_embed1[]; };
layout(set = 0, binding = 5) buffer restrict readonly Embed2Buffer { float t_embed2[]; };
layout(set = 0, binding = 6) buffer restrict readonly Embed3Buffer { float t_embed3[]; };
layout(set = 0, binding = 7) buffer restrict readonly Embed4Buffer { float t_embed4[]; };
layout(set = 0, binding = 8) buffer restrict readonly Embed5Buffer { float t_embed5[]; };
layout(set = 0, binding = 9) buffer restrict readonly Embed6Buffer { float t_embed6[]; };
layout(set = 0, binding = 10) buffer restrict readonly Embed7Buffer { float t_embed7[]; };
layout(set = 0, binding = 11) buffer restrict readonly Weight0Buffer { float t_weight0[]; };
layout(set = 0, binding = 12) buffer restrict readonly Bias0Buffer { float t_bias0[]; };
layout(set = 0, binding = 13) buffer restrict readonly Weight1Buffer { float t_weight1[]; };
layout(set = 0, binding = 14) buffer restrict readonly Bias1Buffer { float t_bias1[]; };
layout(set = 0, binding = 15) buffer restrict readonly Weight2Buffer { float t_weight2[]; };
layout(set = 0, binding = 16) buffer restrict readonly Bias2Buffer { float t_bias2[]; };
layout(set = 0, binding = 17) buffer restrict readonly Weight3Buffer { float t_weight3[]; };
layout(set = 0, binding = 18) buffer restrict readonly Bias3Buffer { float t_bias3[]; };
layout(set = 0, binding = 19) buffer restrict readonly Weight4Buffer { float t_weight4[]; };
layout(set = 0, binding = 20) buffer restrict readonly Bias4Buffer { float t_bias4[]; };
layout(set = 0, binding = 21) buffer restrict readonly Weight5Buffer { float t_weight5[]; };
layout(set = 0, binding = 22) buffer restrict readonly Bias5Buffer { float t_bias5[]; };
layout(set = 0, binding = 23) buffer restrict readonly Weight6Buffer { float t_weight6[]; };
layout(set = 0, binding = 24) buffer restrict readonly Bias6Buffer { float t_bias6[]; };
layout(set = 0, binding = 25) buffer restrict readonly Weight7Buffer { float t_weight7[]; };
layout(set = 0, binding = 26) buffer restrict readonly Bias7Buffer { float t_bias7[]; };
layout(set = 0, binding = 27) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float s_best_score[256];
shared int s_best_id[256];

float read_embed(uint codebook, uint idx) {
    if (codebook == 0u) return t_embed0[idx];
    if (codebook == 1u) return t_embed1[idx];
    if (codebook == 2u) return t_embed2[idx];
    if (codebook == 3u) return t_embed3[idx];
    if (codebook == 4u) return t_embed4[idx];
    if (codebook == 5u) return t_embed5[idx];
    if (codebook == 6u) return t_embed6[idx];
    return t_embed7[idx];
}

float read_weight(uint codebook, uint idx) {
    if (codebook == 0u) return t_weight0[idx];
    if (codebook == 1u) return t_weight1[idx];
    if (codebook == 2u) return t_weight2[idx];
    if (codebook == 3u) return t_weight3[idx];
    if (codebook == 4u) return t_weight4[idx];
    if (codebook == 5u) return t_weight5[idx];
    if (codebook == 6u) return t_weight6[idx];
    return t_weight7[idx];
}

float read_bias(uint codebook, uint idx) {
    if (codebook == 0u) return t_bias0[idx];
    if (codebook == 1u) return t_bias1[idx];
    if (codebook == 2u) return t_bias2[idx];
    if (codebook == 3u) return t_bias3[idx];
    if (codebook == 4u) return t_bias4[idx];
    if (codebook == 5u) return t_bias5[idx];
    if (codebook == 6u) return t_bias6[idx];
    return t_bias7[idx];
}

void main() {
    const uint step = gl_WorkGroupID.x;
    const uint batch = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const uint steps = uint(sizes.x);
    const uint batches = uint(sizes.y);
    const uint dims = uint(sizes.z);
    const uint vocab = uint(sizes.w);
    if (step >= steps || batch >= batches) {
        return;
    }

    const uint ids_base = (batch * 8u) * steps + step;
    const int id0 = clamp(t_audio_ids[ids_base + 0u * steps], 0, int(vocab) - 1);
    const int id1 = clamp(t_audio_ids[ids_base + 1u * steps], 0, int(vocab) - 1);
    const int id2 = clamp(t_audio_ids[ids_base + 2u * steps], 0, int(vocab) - 1);
    const int id3 = clamp(t_audio_ids[ids_base + 3u * steps], 0, int(vocab) - 1);
    const int id4 = clamp(t_audio_ids[ids_base + 4u * steps], 0, int(vocab) - 1);
    const int id5 = clamp(t_audio_ids[ids_base + 5u * steps], 0, int(vocab) - 1);
    const int id6 = clamp(t_audio_ids[ids_base + 6u * steps], 0, int(vocab) - 1);
    const int id7 = clamp(t_audio_ids[ids_base + 7u * steps], 0, int(vocab) - 1);
    const uint u0 = uint(id0);
    const uint u1 = uint(id1);
    const uint u2 = uint(id2);
    const uint u3 = uint(id3);
    const uint u4 = uint(id4);
    const uint u5 = uint(id5);
    const uint u6 = uint(id6);
    const uint u7 = uint(id7);

    float hidden[64];
    for (uint d = 0u; d < dims; ++d) {
        hidden[d] =
            read_embed(0u, u0 * dims + d) + read_embed(1u, u1 * dims + d) +
            read_embed(2u, u2 * dims + d) + read_embed(3u, u3 * dims + d) +
            read_embed(4u, u4 * dims + d) + read_embed(5u, u5 * dims + d) +
            read_embed(6u, u6 * dims + d) + read_embed(7u, u7 * dims + d);
    }

    for (uint c = 0u; c < 8u; ++c) {
        float local_best = -3.402823466e38;
        int local_id = 0;
        for (uint v = lane; v < vocab; v += 256u) {
            float acc = read_bias(c, v);
            const uint w_base = v * dims;
            for (uint d = 0u; d < dims; ++d) {
                acc += read_weight(c, w_base + d) * hidden[d];
            }
            if (acc > local_best) {
                local_best = acc;
                local_id = int(v);
            }
        }
        s_best_score[lane] = local_best;
        s_best_id[lane] = local_id;
        barrier();
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                const float rhs = s_best_score[lane + stride];
                if (rhs > s_best_score[lane]) {
                    s_best_score[lane] = rhs;
                    s_best_id[lane] = s_best_id[lane + stride];
                }
            }
            barrier();
        }
        if (lane == 0u) {
            const uint out_idx = (batch * 8u + c) * steps + step;
            t_output_ids[out_idx] = s_best_id[0];
            t_output_scores[out_idx] = s_best_score[0];
        }
        barrier();
    }
}
"""


OMNIVOICE_STAGE1_QUANTIZER_EMBED_PROJECT_OUT_ARGMAX_ALL_CODEBOOKS_F32 = ShaderVariant(
    name="omnivoice_stage1_quantizer_embed_project_out_argmax_all_codebooks_f32",
    family="omnivoice_stage1",
    contract=ShaderContract(
        name="omnivoice_stage1_quantizer_embed_project_out_argmax_all_codebooks_f32",
        inputs={
            "audio_ids": TensorContract(dtype="int32", shape=("B", "C", "S")),
            "embed0": TensorContract(dtype="float32", shape=("V", "D")),
            "embed1": TensorContract(dtype="float32", shape=("V", "D")),
            "embed2": TensorContract(dtype="float32", shape=("V", "D")),
            "embed3": TensorContract(dtype="float32", shape=("V", "D")),
            "embed4": TensorContract(dtype="float32", shape=("V", "D")),
            "embed5": TensorContract(dtype="float32", shape=("V", "D")),
            "embed6": TensorContract(dtype="float32", shape=("V", "D")),
            "embed7": TensorContract(dtype="float32", shape=("V", "D")),
            "weight0": TensorContract(dtype="float32", shape=("V", "D")),
            "bias0": TensorContract(dtype="float32", shape=("V",)),
            "weight1": TensorContract(dtype="float32", shape=("V", "D")),
            "bias1": TensorContract(dtype="float32", shape=("V",)),
            "weight2": TensorContract(dtype="float32", shape=("V", "D")),
            "bias2": TensorContract(dtype="float32", shape=("V",)),
            "weight3": TensorContract(dtype="float32", shape=("V", "D")),
            "bias3": TensorContract(dtype="float32", shape=("V",)),
            "weight4": TensorContract(dtype="float32", shape=("V", "D")),
            "bias4": TensorContract(dtype="float32", shape=("V",)),
            "weight5": TensorContract(dtype="float32", shape=("V", "D")),
            "bias5": TensorContract(dtype="float32", shape=("V",)),
            "weight6": TensorContract(dtype="float32", shape=("V", "D")),
            "bias6": TensorContract(dtype="float32", shape=("V",)),
            "weight7": TensorContract(dtype="float32", shape=("V", "D")),
            "bias7": TensorContract(dtype="float32", shape=("V",)),
        },
        outputs={
            "output_ids": TensorContract(dtype="int32", shape=("B", "C", "S")),
            "output_scores": TensorContract(dtype="float32", shape=("B", "C", "S")),
        },
        bindings=(
            Binding("output_ids", 0, BindingAccess.WRITE),
            Binding("output_scores", 1, BindingAccess.WRITE),
            Binding("audio_ids", 2, BindingAccess.READ),
            Binding("embed0", 3, BindingAccess.READ),
            Binding("embed1", 4, BindingAccess.READ),
            Binding("embed2", 5, BindingAccess.READ),
            Binding("embed3", 6, BindingAccess.READ),
            Binding("embed4", 7, BindingAccess.READ),
            Binding("embed5", 8, BindingAccess.READ),
            Binding("embed6", 9, BindingAccess.READ),
            Binding("embed7", 10, BindingAccess.READ),
            Binding("weight0", 11, BindingAccess.READ),
            Binding("bias0", 12, BindingAccess.READ),
            Binding("weight1", 13, BindingAccess.READ),
            Binding("bias1", 14, BindingAccess.READ),
            Binding("weight2", 15, BindingAccess.READ),
            Binding("bias2", 16, BindingAccess.READ),
            Binding("weight3", 17, BindingAccess.READ),
            Binding("bias3", 18, BindingAccess.READ),
            Binding("weight4", 19, BindingAccess.READ),
            Binding("bias4", 20, BindingAccess.READ),
            Binding("weight5", 21, BindingAccess.READ),
            Binding("bias5", 22, BindingAccess.READ),
            Binding("weight6", 23, BindingAccess.READ),
            Binding("bias6", 24, BindingAccess.READ),
            Binding("weight7", 25, BindingAccess.READ),
            Binding("bias7", 26, BindingAccess.READ),
        ),
        uniforms=(UniformBlock("sizes", 27, ("S", "B", "D", "V")),),
        dispatch=("S", "B", 1),
        push_constants=None,
    ),
    source=_SOURCE,
)
