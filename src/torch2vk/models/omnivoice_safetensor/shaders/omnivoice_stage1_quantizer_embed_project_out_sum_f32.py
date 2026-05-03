"""Omnivoice Stage1 Quantizer Embed Project Out Sum F32."""

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
layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { float t_output[]; };
layout(set = 0, binding = 1) buffer restrict readonly AudioIdsBuffer { int t_audio_ids[]; };
layout(set = 0, binding = 2) buffer restrict readonly Embed0Buffer { float t_embed0[]; };
layout(set = 0, binding = 3) buffer restrict readonly Embed1Buffer { float t_embed1[]; };
layout(set = 0, binding = 4) buffer restrict readonly Embed2Buffer { float t_embed2[]; };
layout(set = 0, binding = 5) buffer restrict readonly Embed3Buffer { float t_embed3[]; };
layout(set = 0, binding = 6) buffer restrict readonly Embed4Buffer { float t_embed4[]; };
layout(set = 0, binding = 7) buffer restrict readonly Embed5Buffer { float t_embed5[]; };
layout(set = 0, binding = 8) buffer restrict readonly Embed6Buffer { float t_embed6[]; };
layout(set = 0, binding = 9) buffer restrict readonly Embed7Buffer { float t_embed7[]; };
layout(set = 0, binding = 10) buffer restrict readonly Weight0Buffer { float t_weight0[]; };
layout(set = 0, binding = 11) buffer restrict readonly Bias0Buffer { float t_bias0[]; };
layout(set = 0, binding = 12) buffer restrict readonly Weight1Buffer { float t_weight1[]; };
layout(set = 0, binding = 13) buffer restrict readonly Bias1Buffer { float t_bias1[]; };
layout(set = 0, binding = 14) buffer restrict readonly Weight2Buffer { float t_weight2[]; };
layout(set = 0, binding = 15) buffer restrict readonly Bias2Buffer { float t_bias2[]; };
layout(set = 0, binding = 16) buffer restrict readonly Weight3Buffer { float t_weight3[]; };
layout(set = 0, binding = 17) buffer restrict readonly Bias3Buffer { float t_bias3[]; };
layout(set = 0, binding = 18) buffer restrict readonly Weight4Buffer { float t_weight4[]; };
layout(set = 0, binding = 19) buffer restrict readonly Bias4Buffer { float t_bias4[]; };
layout(set = 0, binding = 20) buffer restrict readonly Weight5Buffer { float t_weight5[]; };
layout(set = 0, binding = 21) buffer restrict readonly Bias5Buffer { float t_bias5[]; };
layout(set = 0, binding = 22) buffer restrict readonly Weight6Buffer { float t_weight6[]; };
layout(set = 0, binding = 23) buffer restrict readonly Bias6Buffer { float t_bias6[]; };
layout(set = 0, binding = 24) buffer restrict readonly Weight7Buffer { float t_weight7[]; };
layout(set = 0, binding = 25) buffer restrict readonly Bias7Buffer { float t_bias7[]; };
layout(set = 0, binding = 26) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint h = gl_GlobalInvocationID.x;
    const uint t = gl_GlobalInvocationID.y;
    const uint b = gl_GlobalInvocationID.z;
    const uint steps = uint(sizes.x);
    const uint batches = uint(sizes.y);
    if (h >= 1024u || t >= steps || b >= batches) {
        return;
    }
    const uint base = (b * 8u + 0u) * steps + t;
    const int id0 = clamp(t_audio_ids[base], 0, 1023);
    const int id1 = clamp(t_audio_ids[base + steps], 0, 1023);
    const int id2 = clamp(t_audio_ids[base + 2u * steps], 0, 1023);
    const int id3 = clamp(t_audio_ids[base + 3u * steps], 0, 1023);
    const int id4 = clamp(t_audio_ids[base + 4u * steps], 0, 1023);
    const int id5 = clamp(t_audio_ids[base + 5u * steps], 0, 1023);
    const int id6 = clamp(t_audio_ids[base + 6u * steps], 0, 1023);
    const int id7 = clamp(t_audio_ids[base + 7u * steps], 0, 1023);
    const uint w_base = h * 64u;
    const uint e0 = uint(id0) * 64u;
    const uint e1 = uint(id1) * 64u;
    const uint e2 = uint(id2) * 64u;
    const uint e3 = uint(id3) * 64u;
    const uint e4 = uint(id4) * 64u;
    const uint e5 = uint(id5) * 64u;
    const uint e6 = uint(id6) * 64u;
    const uint e7 = uint(id7) * 64u;
    float outv =
        t_bias0[h] + t_bias1[h] + t_bias2[h] + t_bias3[h] +
        t_bias4[h] + t_bias5[h] + t_bias6[h] + t_bias7[h];
    for (uint d = 0u; d < 64u; ++d) {
        const uint wi = w_base + d;
        outv += t_embed0[e0 + d] * t_weight0[wi];
        outv += t_embed1[e1 + d] * t_weight1[wi];
        outv += t_embed2[e2 + d] * t_weight2[wi];
        outv += t_embed3[e3 + d] * t_weight3[wi];
        outv += t_embed4[e4 + d] * t_weight4[wi];
        outv += t_embed5[e5 + d] * t_weight5[wi];
        outv += t_embed6[e6 + d] * t_weight6[wi];
        outv += t_embed7[e7 + d] * t_weight7[wi];
    }
    const uint out_idx = (b * steps + t) * 1024u + h;
    t_output[out_idx] = outv;
}
"""


OMNIVOICE_STAGE1_QUANTIZER_EMBED_PROJECT_OUT_SUM_F32 = ShaderVariant(
    name="omnivoice_stage1_quantizer_embed_project_out_sum_f32",
    family="omnivoice_stage1",
    contract=ShaderContract(
        name="omnivoice_stage1_quantizer_embed_project_out_sum_f32",
        inputs={
            "audio_ids": TensorContract(dtype="int32", shape=("B", 8, "S")),
            "embed0": TensorContract(dtype="float32", shape=(1024, 64)),
            "embed1": TensorContract(dtype="float32", shape=(1024, 64)),
            "embed2": TensorContract(dtype="float32", shape=(1024, 64)),
            "embed3": TensorContract(dtype="float32", shape=(1024, 64)),
            "embed4": TensorContract(dtype="float32", shape=(1024, 64)),
            "embed5": TensorContract(dtype="float32", shape=(1024, 64)),
            "embed6": TensorContract(dtype="float32", shape=(1024, 64)),
            "embed7": TensorContract(dtype="float32", shape=(1024, 64)),
            "weight0": TensorContract(dtype="float32", shape=(1024, 64)),
            "bias0": TensorContract(dtype="float32", shape=(1024,)),
            "weight1": TensorContract(dtype="float32", shape=(1024, 64)),
            "bias1": TensorContract(dtype="float32", shape=(1024,)),
            "weight2": TensorContract(dtype="float32", shape=(1024, 64)),
            "bias2": TensorContract(dtype="float32", shape=(1024,)),
            "weight3": TensorContract(dtype="float32", shape=(1024, 64)),
            "bias3": TensorContract(dtype="float32", shape=(1024,)),
            "weight4": TensorContract(dtype="float32", shape=(1024, 64)),
            "bias4": TensorContract(dtype="float32", shape=(1024,)),
            "weight5": TensorContract(dtype="float32", shape=(1024, 64)),
            "bias5": TensorContract(dtype="float32", shape=(1024,)),
            "weight6": TensorContract(dtype="float32", shape=(1024, 64)),
            "bias6": TensorContract(dtype="float32", shape=(1024,)),
            "weight7": TensorContract(dtype="float32", shape=(1024, 64)),
            "bias7": TensorContract(dtype="float32", shape=(1024,)),
        },
        outputs={
            "output": TensorContract(dtype="float32", shape=("B", "S", 1024)),
        },
        bindings=(
            Binding("output", 0, BindingAccess.WRITE),
            Binding("audio_ids", 1, BindingAccess.READ),
            Binding("embed0", 2, BindingAccess.READ),
            Binding("embed1", 3, BindingAccess.READ),
            Binding("embed2", 4, BindingAccess.READ),
            Binding("embed3", 5, BindingAccess.READ),
            Binding("embed4", 6, BindingAccess.READ),
            Binding("embed5", 7, BindingAccess.READ),
            Binding("embed6", 8, BindingAccess.READ),
            Binding("embed7", 9, BindingAccess.READ),
            Binding("weight0", 10, BindingAccess.READ),
            Binding("bias0", 11, BindingAccess.READ),
            Binding("weight1", 12, BindingAccess.READ),
            Binding("bias1", 13, BindingAccess.READ),
            Binding("weight2", 14, BindingAccess.READ),
            Binding("bias2", 15, BindingAccess.READ),
            Binding("weight3", 16, BindingAccess.READ),
            Binding("bias3", 17, BindingAccess.READ),
            Binding("weight4", 18, BindingAccess.READ),
            Binding("bias4", 19, BindingAccess.READ),
            Binding("weight5", 20, BindingAccess.READ),
            Binding("bias5", 21, BindingAccess.READ),
            Binding("weight6", 22, BindingAccess.READ),
            Binding("bias6", 23, BindingAccess.READ),
            Binding("weight7", 24, BindingAccess.READ),
            Binding("bias7", 25, BindingAccess.READ),
        ),
        uniforms=(UniformBlock("sizes", 26, ("S", "B", 64, 1024)),),
        dispatch=("((1024) + (128) - 1)//(128)", "S", "B"),
        push_constants=None,
    ),
    source=_SOURCE,
)
