"""Omnivoice Stage1 Snake Conv1D K1 Residual Add F32."""

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
layout(set = 0, binding = 1) buffer restrict readonly XBuffer { float t_x[]; };
layout(set = 0, binding = 2) buffer restrict readonly AlphaBuffer { float t_alpha[]; };
layout(set = 0, binding = 3) buffer restrict readonly WeightBuffer { float t_weight[]; };
layout(set = 0, binding = 4) buffer restrict readonly BiasBuffer { float t_bias[]; };
layout(set = 0, binding = 5) buffer restrict readonly ResidualBuffer { float t_residual[]; };
layout(set = 0, binding = 6) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

float snake(float v, float a) {
    const float aa = a + 1.0e-9;
    const float s = sin(a * v);
    return v + (s * s) / aa;
}

void main() {
    const uint oc = gl_GlobalInvocationID.x;
    const uint t = gl_GlobalInvocationID.y;
    const uint b = gl_GlobalInvocationID.z;
    const uint steps = uint(sizes.x);
    const uint batches = uint(sizes.y);
    const uint in_channels = uint(sizes.z);
    const uint out_channels = uint(sizes.w);
    if (oc >= out_channels || t >= steps || b >= batches) {
        return;
    }
    float acc = t_bias[oc];
    for (uint ic = 0u; ic < in_channels; ++ic) {
        const uint x_idx = (b * steps + t) * in_channels + ic;
        const uint w_idx = oc * in_channels + ic;
        acc += snake(t_x[x_idx], t_alpha[ic]) * t_weight[w_idx];
    }
    const uint out_idx = (b * steps + t) * out_channels + oc;
    t_output[out_idx] = acc + t_residual[out_idx];
}
"""


OMNIVOICE_STAGE1_SNAKE_CONV1D_K1_RESIDUAL_ADD_F32 = ShaderVariant(
    name="omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
    family="omnivoice_stage1",
    contract=ShaderContract(
        name="omnivoice_stage1_snake_conv1d_k1_residual_add_f32",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "S", "IC")),
            "alpha": TensorContract(dtype="float32", shape=(1, "A", 1)),
            "weight": TensorContract(dtype="float32", shape=("OC", "IC", 1)),
            "bias": TensorContract(dtype="float32", shape=("OC",)),
            "residual": TensorContract(dtype="float32", shape=("B", "S", "OC")),
        },
        outputs={
            "output": TensorContract(dtype="float32", shape=("B", "S", "OC")),
        },
        bindings=(
            Binding("output", 0, BindingAccess.WRITE),
            Binding("x", 1, BindingAccess.READ),
            Binding("alpha", 2, BindingAccess.READ),
            Binding("weight", 3, BindingAccess.READ),
            Binding("bias", 4, BindingAccess.READ),
            Binding("residual", 5, BindingAccess.READ),
        ),
        uniforms=(UniformBlock("sizes", 6, ("S", "B", "IC", "OC")),),
        dispatch=("((OC) + (128) - 1)//(128)", "S", "B"),
        push_constants=None,
    ),
    source=_SOURCE,
)
