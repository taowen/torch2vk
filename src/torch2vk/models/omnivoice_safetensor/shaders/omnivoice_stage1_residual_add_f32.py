"""Omnivoice Stage1 Residual Add F32."""

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
layout(set = 0, binding = 2) buffer restrict readonly ResidualBuffer { float t_residual[]; };
layout(set = 0, binding = 3) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint c = gl_GlobalInvocationID.x;
    const uint t = gl_GlobalInvocationID.y;
    const uint b = gl_GlobalInvocationID.z;
    const uint steps = uint(sizes.x);
    const uint batches = uint(sizes.y);
    const uint channels = uint(sizes.z);
    if (c >= channels || t >= steps || b >= batches) {
        return;
    }
    const uint idx = (b * steps + t) * channels + c;
    t_output[idx] = t_x[idx] + t_residual[idx];
}
"""


OMNIVOICE_STAGE1_RESIDUAL_ADD_F32 = ShaderVariant(
    name="omnivoice_stage1_residual_add_f32",
    family="omnivoice_stage1",
    contract=ShaderContract(
        name="omnivoice_stage1_residual_add_f32",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "S", "C")),
            "residual": TensorContract(dtype="float32", shape=("B", "S", "C")),
        },
        outputs={
            "output": TensorContract(dtype="float32", shape=("B", "S", "C")),
        },
        bindings=(
            Binding("output", 0, BindingAccess.WRITE),
            Binding("x", 1, BindingAccess.READ),
            Binding("residual", 2, BindingAccess.READ),
        ),
        uniforms=(UniformBlock("sizes", 3, ("S", "B", "C", 1)),),
        dispatch=("((C) + (256) - 1)//(256)", "S", "B"),
        push_constants=None,
    ),
    source=_SOURCE,
)
