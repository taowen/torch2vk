"""Qwen3 residual add shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_shader_variant_source
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

ADD_F32 = ShaderVariant(
    name="add_f32_f32_f32_norepeat",
    family="add",
    contract=ShaderContract(
        name="add_f32_f32_f32_norepeat",
        inputs={
            "lhs": TensorContract(dtype="float32", shape=("B", "S", "H")),
            "rhs": TensorContract(dtype="float32", shape=("B", "S", "H")),
        },
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "H"))},
        bindings=(
            Binding("lhs", 0, BindingAccess.READ),
            Binding("rhs", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("H", "S", "B"),
        resources=(ResourceBinding("partial_buffer", 3, "storage_buffer"),),
        push_constants=PushConstantBlock(
            size=116,
            fields=(
                PushConstantField("ne", 0, "uint32", "output.numel"),
                PushConstantField("src0_ne0", 4, "uint32", "lhs.dim2"),
                PushConstantField("src0_ne1", 8, "uint32", "lhs.dim1"),
                PushConstantField("src0_ne2", 12, "uint32", "lhs.dim0"),
                PushConstantField("src0_ne3", 16, "uint32", 1),
                PushConstantField("src0_nb0", 20, "uint32", 1),
                PushConstantField("src0_nb1", 24, "uint32", "lhs.dim2"),
                PushConstantField("src0_nb2", 28, "uint32", "lhs.dim2*lhs.dim1"),
                PushConstantField("src0_nb3", 32, "uint32", "lhs.numel"),
                PushConstantField("src1_ne0", 36, "uint32", "rhs.dim2"),
                PushConstantField("src1_ne1", 40, "uint32", "rhs.dim1"),
                PushConstantField("src1_ne2", 44, "uint32", "rhs.dim0"),
                PushConstantField("src1_ne3", 48, "uint32", 1),
                PushConstantField("src1_nb0", 52, "uint32", 1),
                PushConstantField("src1_nb1", 56, "uint32", "rhs.dim2"),
                PushConstantField("src1_nb2", 60, "uint32", "rhs.dim2*rhs.dim1"),
                PushConstantField("src1_nb3", 64, "uint32", "rhs.numel"),
                PushConstantField("dst_ne0", 68, "uint32", "output.dim2"),
                PushConstantField("dst_ne1", 72, "uint32", "output.dim1"),
                PushConstantField("dst_ne2", 76, "uint32", "output.dim0"),
                PushConstantField("dst_ne3", 80, "uint32", 1),
                PushConstantField("dst_nb0", 84, "uint32", 1),
                PushConstantField("dst_nb1", 88, "uint32", "output.dim2"),
                PushConstantField("dst_nb2", 92, "uint32", "output.dim2*output.dim1"),
                PushConstantField("dst_nb3", 96, "uint32", "output.numel"),
                PushConstantField("misalign_offsets", 100, "uint32", 0),
                PushConstantField("param1", 104, "float32", 0.0),
                PushConstantField("param2", 108, "float32", 0.0),
                PushConstantField("param3", 112, "int32", 0),
            ),
        ),
    ),
    source=copied_shader_variant_source(
        "add_f32_f32_f32_norepeat.py",
        "ADD_F32_F32_F32_NOREPEAT",
    ),
)
