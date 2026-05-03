"""Omnivoice Stage1 Snake Deconv1D Block0 F32."""

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
layout(set = 0, binding = 5) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

float snake(float v, float a) {
    const float aa = a + 1.0e-9;
    const float s = sin(a * v);
    return v + (s * s) / aa;
}

void main() {
    const uint oc = gl_GlobalInvocationID.x;
    const uint to = gl_GlobalInvocationID.y;
    const uint b = gl_GlobalInvocationID.z;
    const uint si = uint(sizes.x);
    const uint so = uint(sizes.y);
    const uint ic = uint(sizes.z);
    const uint ocn = uint(sizes.w);
    if (oc >= ocn || to >= so) {
        return;
    }
    const uint alpha_len = uint(t_alpha.length());
    float acc = t_bias[oc];
    for (uint ii = 0u; ii < ic; ++ii) {
        const float a = t_alpha[ii % alpha_len];
        for (uint k = 0u; k < 16u; ++k) {
            const int numer = int(to) + 4 - int(k);
            if ((numer & 7) != 0) {
                continue;
            }
            const int ti = numer >> 3;
            if (ti < 0 || ti >= int(si)) {
                continue;
            }
            const uint x_idx = (b * si + uint(ti)) * ic + ii;
            const uint w_idx = (ii * ocn + oc) * 16u + k;
            acc += snake(t_x[x_idx], a) * t_weight[w_idx];
        }
    }
    const uint out_idx = (b * so + to) * ocn + oc;
    t_output[out_idx] = acc;
}
"""


OMNIVOICE_STAGE1_SNAKE_DECONV1D_BLOCK0_F32 = ShaderVariant(
    name="omnivoice_stage1_snake_deconv1d_block0_f32",
    family="omnivoice_stage1",
    contract=ShaderContract(
        name="omnivoice_stage1_snake_deconv1d_block0_f32",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "SI", "IC")),
            "alpha": TensorContract(dtype="float32", shape=(1, "A", 1)),
            "weight": TensorContract(dtype="float32", shape=("IC", "OC", 16)),
            "bias": TensorContract(dtype="float32", shape=("OC",)),
        },
        outputs={
            "output": TensorContract(dtype="float32", shape=("B", "SO", "OC")),
        },
        bindings=(
            Binding("output", 0, BindingAccess.WRITE),
            Binding("x", 1, BindingAccess.READ),
            Binding("alpha", 2, BindingAccess.READ),
            Binding("weight", 3, BindingAccess.READ),
            Binding("bias", 4, BindingAccess.READ),
        ),
        uniforms=(UniformBlock("sizes", 5, ("SI", "SO", "IC", "OC")),),
        dispatch=("((OC) + (128) - 1)//(128)", "SO", "B"),
        push_constants=None,
    ),
    source=_SOURCE,
)
