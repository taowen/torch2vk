"""Omnivoice Stage1 Quantizer Embed Sum F32."""

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
layout(set = 0, binding = 10) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint d = gl_GlobalInvocationID.x;
    const uint step = gl_GlobalInvocationID.y;
    const uint batch = gl_GlobalInvocationID.z;
    const uint steps = uint(sizes.x);
    const uint batches = uint(sizes.y);
    const uint dims = uint(sizes.z);
    const uint vocab = uint(sizes.w);
    if (d >= dims || step >= steps || batch >= batches) {
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
    const uint out_idx = (batch * steps + step) * dims + d;
    const uint o0 = uint(id0) * dims + d;
    const uint o1 = uint(id1) * dims + d;
    const uint o2 = uint(id2) * dims + d;
    const uint o3 = uint(id3) * dims + d;
    const uint o4 = uint(id4) * dims + d;
    const uint o5 = uint(id5) * dims + d;
    const uint o6 = uint(id6) * dims + d;
    const uint o7 = uint(id7) * dims + d;
    t_output[out_idx] = t_embed0[o0] + t_embed1[o1] + t_embed2[o2] + t_embed3[o3] + t_embed4[o4] + t_embed5[o5] + t_embed6[o6] + t_embed7[o7];
}
"""


OMNIVOICE_STAGE1_QUANTIZER_EMBED_SUM_F32 = ShaderVariant(
    name="omnivoice_stage1_quantizer_embed_sum_f32",
    family="omnivoice_stage1",
    contract=ShaderContract(
        name="omnivoice_stage1_quantizer_embed_sum_f32",
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
        },
        outputs={
            "output": TensorContract(dtype="float32", shape=("B", "S", "D")),
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
        ),
        uniforms=(UniformBlock("sizes", 10, ("S", "B", "D", "V")),),
        dispatch=("((D) + (256) - 1)//(256)", "S", "B"),
        push_constants=None,
    ),
    source=_SOURCE,
)
