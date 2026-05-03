"""Qwen3 BF16 linear shader."""

from __future__ import annotations

from torch2vk.shader import (
    Binding,
    BindingAccess,
    PushConstantBlock,
    PushConstantField,
    ResourceBinding,
    ShaderContract,
    ShaderVariant,
    TensorContract,
)

_SOURCE = """
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer A { uint16_t data_a[]; };
layout(binding = 1) readonly buffer B { float data_b[]; };
layout(binding = 2) buffer D { float data_d[]; };
layout(binding = 3) readonly buffer F0 { float data_f0[]; };
layout(binding = 4) readonly buffer F1 { float data_f1[]; };

layout(push_constant) uniform PushConstants {
    uint ncols;
    uint stride_a;
    uint stride_b;
    uint stride_d;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;
    uint fusion_flags;
    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
} p;

float bf16_to_fp32(uint16_t bits) {
    return uintBitsToFloat(uint(bits) << 16);
}

void main() {
    uint row = gl_WorkGroupID.x;
    uint batch = gl_WorkGroupID.y;
    uint step = gl_WorkGroupID.z;

    uint nrows = p.stride_d;
    if (row >= nrows) {
        return;
    }

    uint a_base = row * p.stride_a;
    uint x_base = batch * p.batch_stride_b + step * p.stride_b;
    uint out_idx = batch * p.batch_stride_d + step * p.stride_d + row;

    float acc = 0.0;
    for (uint k = 0; k < p.ncols; ++k) {
        float w = bf16_to_fp32(data_a[a_base + k]);
        float x = data_b[x_base + k];
        acc += w * x;
    }

    if ((p.fusion_flags & 1u) != 0u) {
        acc += data_f0[out_idx];
    }
    data_d[out_idx] = acc;
}
"""

LINEAR_BF16_F32 = ShaderVariant(
    name="linear_bf16_f32",
    family="linear",
    contract=ShaderContract(
        name="linear_bf16_f32",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "S", "K")),
            "weight": TensorContract(dtype="bfloat16", shape=("N", "K")),
        },
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "N"))},
        bindings=(
            Binding("weight", 0, BindingAccess.READ),
            Binding("x", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        resources=(
            ResourceBinding("fuse0_placeholder", 3, "storage_buffer"),
            ResourceBinding("fuse1_placeholder", 4, "storage_buffer"),
        ),
        dispatch=("N", "S", "B"),
        push_constants=PushConstantBlock(
            size=52,
            fields=(
                PushConstantField("ncols", 0, "uint32", "K"),
                PushConstantField("stride_a", 4, "uint32", "K"),
                PushConstantField("stride_b", 8, "uint32", "K"),
                PushConstantField("stride_d", 12, "uint32", "N"),
                PushConstantField("batch_stride_a", 16, "uint32", "N*K"),
                PushConstantField("batch_stride_b", 20, "uint32", "S*K"),
                PushConstantField("batch_stride_d", 24, "uint32", "S*N"),
                PushConstantField("fusion_flags", 28, "uint32", 0),
                PushConstantField("base_work_group_y", 32, "uint32", 0),
                PushConstantField("ne02", 36, "uint32", 1),
                PushConstantField("ne12", 40, "uint32", "B"),
                PushConstantField("broadcast2", 44, "uint32", 1),
                PushConstantField("broadcast3", 48, "uint32", 1),
            ),
        ),
    ),
    source=_SOURCE,
)
