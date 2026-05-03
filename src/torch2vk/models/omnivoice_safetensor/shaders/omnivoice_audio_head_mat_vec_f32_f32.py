"""OmniVoice audio head float32 matvec shader."""

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

layout(set = 0, binding = 0) buffer restrict readonly WeightBuffer {
    float t_weight[];
};

layout(set = 0, binding = 1) buffer restrict readonly XBuffer {
    float t_x[];
};

layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer {
    float t_output[];
};

layout(set = 0, binding = 3) uniform restrict readonly sizes_UBO {
    ivec4 sizes;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float partial[256];

void main() {
    const uint vocab_index = gl_WorkGroupID.x;
    const uint step_index = gl_WorkGroupID.y;
    const uint batch_index = gl_WorkGroupID.z;
    const uint lane = gl_LocalInvocationID.x;

    const uint hidden = uint(sizes.x);
    const uint steps = uint(sizes.y);
    const uint batches = uint(sizes.z);
    const uint vocab = uint(sizes.w);
    if (vocab_index >= vocab || step_index >= steps || batch_index >= batches) {
        return;
    }

    const uint x_base = (batch_index * steps + step_index) * hidden;
    const uint w_base = vocab_index * hidden;
    float acc = 0.0;
    for (uint hidden_index = lane; hidden_index < hidden; hidden_index += 256u) {
        acc += t_weight[w_base + hidden_index] * t_x[x_base + hidden_index];
    }
    partial[lane] = acc;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        barrier();
    }

    if (lane == 0u) {
        t_output[(batch_index * steps + step_index) * vocab + vocab_index] = partial[0];
    }
}
"""


OMNIVOICE_AUDIO_HEAD_MAT_VEC_F32_F32 = ShaderVariant(
    name="omnivoice_audio_head_mat_vec_f32_f32",
    family="omnivoice_audio_head_mat_vec",
    contract=ShaderContract(
        name="omnivoice_audio_head_mat_vec_f32_f32",
        inputs={
            "weight": TensorContract(dtype="float32", shape=("V", "H")),
            "x": TensorContract(dtype="float32", shape=("B", "S", "H")),
        },
        outputs={
            "output": TensorContract(dtype="float32", shape=("B", "S", "V")),
        },
        bindings=(
            Binding("weight", 0, BindingAccess.READ),
            Binding("x", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        uniforms=(UniformBlock("sizes", 3, ("H", "S", "B", "V")),),
        dispatch=("V", "S", "B"),
        push_constants=None,
    ),
    source=_SOURCE,
)
