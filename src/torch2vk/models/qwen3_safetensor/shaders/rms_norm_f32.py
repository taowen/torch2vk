"""Qwen3 RMS norm shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_assignment_string
from torch2vk.shader import (
    Binding,
    BindingAccess,
    PushConstantBlock,
    PushConstantField,
    ShaderContract,
    ShaderVariant,
    TensorContract,
)

_SOURCE = (
    copied_assignment_string(
        "rms_norm_f32_f32_weight_llama_wg512.py",
        "_RMS_NORM_COMP_SOURCE",
    )
    .replace(
        "layout(binding = 1) readonly buffer B { float data_b[]; };",
        "layout(binding = 1) readonly buffer B { uint16_t data_b[]; };",
    )
    .replace(
        "float(data_b[b_offset + fastmod(col, p.ne10)])",
        "bf16_to_fp32(uint(data_b[b_offset + fastmod(col, p.ne10)]))",
    )
    .replace(
        "float(data_b[b_offset + col])",
        "bf16_to_fp32(uint(data_b[b_offset + col]))",
    )
)

RMS_NORM_F32 = ShaderVariant(
    name="rms_norm_f32_bf16_weight_llama_wg512",
    family="rms_norm",
    contract=ShaderContract(
        name="rms_norm_f32_bf16_weight_llama_wg512",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "S", "H")),
            "weight": TensorContract(dtype="bfloat16", shape=("H",)),
        },
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "H"))},
        bindings=(
            Binding("x", 0, BindingAccess.READ),
            Binding("weight", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("S", "B", 1),
        push_constants=PushConstantBlock(
            size=116,
            fields=(
                PushConstantField("ne", 0, "uint32", "output.numel"),
                PushConstantField("src0_ne0", 4, "uint32", "x.dim2"),
                PushConstantField("src0_ne1", 8, "uint32", "x.dim1"),
                PushConstantField("src0_ne2", 12, "uint32", "x.dim0"),
                PushConstantField("src0_ne3", 16, "uint32", 1),
                PushConstantField("src0_nb0", 20, "uint32", 1),
                PushConstantField("src0_nb1", 24, "uint32", "x.dim2"),
                PushConstantField("src0_nb2", 28, "uint32", "x.dim2*x.dim1"),
                PushConstantField("src0_nb3", 32, "uint32", "x.numel"),
                PushConstantField("src1_ne0", 36, "uint32", "weight.dim0"),
                PushConstantField("src1_ne1", 40, "uint32", 1),
                PushConstantField("src1_ne2", 44, "uint32", 1),
                PushConstantField("src1_ne3", 48, "uint32", 1),
                PushConstantField("src1_nb0", 52, "uint32", 1),
                PushConstantField("src1_nb1", 56, "uint32", "weight.dim0"),
                PushConstantField("src1_nb2", 60, "uint32", "weight.dim0"),
                PushConstantField("src1_nb3", 64, "uint32", "weight.dim0"),
                PushConstantField("dst_ne0", 68, "uint32", "output.dim2"),
                PushConstantField("dst_ne1", 72, "uint32", "output.dim1"),
                PushConstantField("dst_ne2", 76, "uint32", "output.dim0"),
                PushConstantField("dst_ne3", 80, "uint32", 1),
                PushConstantField("dst_nb0", 84, "uint32", 1),
                PushConstantField("dst_nb1", 88, "uint32", "output.dim2"),
                PushConstantField("dst_nb2", 92, "uint32", "output.dim2*output.dim1"),
                PushConstantField("dst_nb3", 96, "uint32", "output.numel"),
                PushConstantField("misalign_offsets", 100, "uint32", 0),
                PushConstantField("param1", 104, "float32", 1.0e-6),
                PushConstantField("param2", 108, "float32", 0.0),
                PushConstantField("param3", 112, "int32", 0),
            ),
        ),
    ),
    specialization_constants={0: 0, 1: 1},
    source=_SOURCE,
)
