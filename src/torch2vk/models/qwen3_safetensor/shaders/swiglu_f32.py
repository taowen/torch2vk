"""Qwen3 SwiGLU shader."""

from __future__ import annotations

from torch2vk.shader import (
    Binding,
    BindingAccess,
    PushConstantBlock,
    PushConstantField,
    ShaderContract,
    ShaderVariant,
    TensorContract,
)

_SOURCE = """
#version 450

#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer A { float data_a[]; };
layout(binding = 1) readonly buffer B { float data_b[]; };
layout(binding = 2) writeonly buffer D { float data_d[]; };

layout(push_constant) uniform parameter
{
    uint N;
    uint ne00;
    uint ne20;
    uint mode;
    float alpha;
    float limit;
    uint nb01;
    uint nb02;
    uint nb03;
    uint ne01;
    uint ne02;
    uint nb11;
    uint nb12;
    uint nb13;
    uint ne11;
    uint ne12;
} p;

float op(float a, float b) {
    return a / (1.0f + exp(- a)) * b;
}

 void main() {
    const uint i = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;

    if (i >= p.N) {
        return;
    }

    const uint row = i / p.ne20;
    const uint col = i - row * p.ne20;

    const uint i3 = row / (p.ne01 * p.ne02);
    const uint i2 = (row % (p.ne01 * p.ne02)) / p.ne01;
    const uint i1 = row % p.ne01;
    const uint src_idx = i3 * p.nb03 + i2 * p.nb02 + i1 * p.nb01 + col;

    const uint dst_i3 = row / (p.ne11 * p.ne12);
    const uint dst_i2 = (row % (p.ne11 * p.ne12)) / p.ne11;
    const uint dst_i1 = row % p.ne11;
    const uint dst_idx = dst_i3 * p.nb13 + dst_i2 * p.nb12 + dst_i1 * p.nb11 + col;

    if (p.mode == 0) {

        const uint offset = p.ne00 / 2;
        const uint idx = src_idx;

        data_d[dst_idx] = float(op(float(data_a[idx]), float(data_a[idx + offset])));
    } else if (p.mode == 1) {

        const uint offset = p.ne00 / 2;
        const uint idx = src_idx;

        data_d[dst_idx] = float(op(float(data_a[idx + offset]), float(data_a[idx])));
    } else {

        const uint idx = src_idx;

        data_d[dst_idx] = float(op(float(data_a[idx]), float(data_b[idx])));
    }
}
"""

SWIGLU_F32 = ShaderVariant(
    name="swiglu_f32",
    family="swiglu",
    contract=ShaderContract(
        name="swiglu_f32",
        inputs={
            "gate": TensorContract(dtype="float32", shape=("B", "S", "I")),
            "up": TensorContract(dtype="float32", shape=("B", "S", "I")),
        },
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "I"))},
        bindings=(
            Binding("gate", 0, BindingAccess.READ),
            Binding("up", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("I", "S", "B"),
        push_constants=PushConstantBlock(
            size=64,
            fields=(
                PushConstantField("N", 0, "uint32", "output.numel"),
                PushConstantField("ne00", 4, "uint32", "gate.dim2"),
                PushConstantField("ne20", 8, "uint32", "output.dim2"),
                PushConstantField("mode", 12, "uint32", 2),
                PushConstantField("alpha", 16, "float32", 0.0),
                PushConstantField("limit", 20, "float32", 0.0),
                PushConstantField("nb01", 24, "uint32", "gate.dim2"),
                PushConstantField("nb02", 28, "uint32", "gate.dim2*gate.dim1"),
                PushConstantField("nb03", 32, "uint32", "gate.numel"),
                PushConstantField("ne01", 36, "uint32", "gate.dim1"),
                PushConstantField("ne02", 40, "uint32", "gate.dim0"),
                PushConstantField("nb11", 44, "uint32", "output.dim2"),
                PushConstantField("nb12", 48, "uint32", "output.dim2*output.dim1"),
                PushConstantField("nb13", 52, "uint32", "output.numel"),
                PushConstantField("ne11", 56, "uint32", "output.dim1"),
                PushConstantField("ne12", 60, "uint32", "output.dim0"),
            ),
        ),
    ),
    source=_SOURCE,
)
