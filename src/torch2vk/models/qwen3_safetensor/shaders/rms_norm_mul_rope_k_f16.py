"""Qwen3 K projection norm, RoPE, and KV-cache write shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_shader_variant_source
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
    copied_shader_variant_source(
        "rms_norm_mul_rope_f32_f16.py",
        "RMS_NORM_MUL_ROPE_F32_F16",
    )
    .replace(
        "layout (binding = 1) readonly buffer B {B_TYPE data_b[];};",
        "layout (binding = 1) readonly buffer B {uint16_t data_b[];};",
    )
    .replace(
        "FLOAT_TYPE(data_b[b_offset + fastmod(col, p.ne10)])",
        "bf16_to_fp32(uint(data_b[b_offset + fastmod(col, p.ne10)]))",
    )
    .replace(
        "FLOAT_TYPE(data_b[b_offset + col])",
        "bf16_to_fp32(uint(data_b[b_offset + col]))",
    )
)

RMS_NORM_MUL_ROPE_K_F16 = ShaderVariant(
    name="rms_norm_mul_rope_f32_bf16_f16",
    family="rms_norm_mul_rope",
    contract=ShaderContract(
        name="rms_norm_mul_rope_f32_bf16_f16",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "S", "N", "D")),
            "weight": TensorContract(dtype="bfloat16", shape=("D",)),
            "position_ids": TensorContract(dtype="int32", shape=("S",)),
            "freq_factors_placeholder": TensorContract(dtype="float32", shape=("D",)),
            "row_indices": TensorContract(dtype="int64", shape=("S",)),
        },
        outputs={"output": TensorContract(dtype="float16", shape=("B", "T", "N", "D"))},
        bindings=(
            Binding("x", 0, BindingAccess.READ),
            Binding("weight", 1, BindingAccess.READ),
            Binding("position_ids", 3, BindingAccess.READ),
            Binding("freq_factors_placeholder", 4, BindingAccess.READ),
            Binding("output", 5, BindingAccess.WRITE),
            Binding("row_indices", 6, BindingAccess.READ),
        ),
        dispatch=("N", "S", "B"),
        push_constants=PushConstantBlock(
            size=224,
            fields=(
                PushConstantField("ne", 0, "uint32", "x.numel"),
                PushConstantField("src0_ne0", 4, "uint32", "D"),
                PushConstantField("src0_ne1", 8, "uint32", "N"),
                PushConstantField("src0_ne2", 12, "uint32", "S"),
                PushConstantField("src0_ne3", 16, "uint32", "B"),
                PushConstantField("src0_nb0", 20, "uint32", 1),
                PushConstantField("src0_nb1", 24, "uint32", "D"),
                PushConstantField("src0_nb2", 28, "uint32", "D*N"),
                PushConstantField("src0_nb3", 32, "uint32", "D*N*S"),
                PushConstantField("src1_ne0", 36, "uint32", "D"),
                PushConstantField("src1_ne1", 40, "uint32", 1),
                PushConstantField("src1_ne2", 44, "uint32", 1),
                PushConstantField("src1_ne3", 48, "uint32", 1),
                PushConstantField("src1_nb0", 52, "uint32", 1),
                PushConstantField("src1_nb1", 56, "uint32", "D"),
                PushConstantField("src1_nb2", 60, "uint32", "D"),
                PushConstantField("src1_nb3", 64, "uint32", "D"),
                PushConstantField("dst_ne0", 68, "uint32", "D"),
                PushConstantField("dst_ne1", 72, "uint32", "N"),
                PushConstantField("dst_ne2", 76, "uint32", "T"),
                PushConstantField("dst_ne3", 80, "uint32", "B"),
                PushConstantField("dst_nb0", 84, "uint32", 1),
                PushConstantField("dst_nb1", 88, "uint32", "D"),
                PushConstantField("dst_nb2", 92, "uint32", "D*N"),
                PushConstantField("dst_nb3", 96, "uint32", "D*N*T"),
                PushConstantField("padding", 100, "uint32", 0),
                PushConstantField("param1", 104, "float32", 1.0e-6),
                PushConstantField("param2", 108, "float32", 0.0),
                PushConstantField("param3", 112, "int32", 0),
                PushConstantField("rope_mode", 116, "uint32", 2),
                PushConstantField("rope_nrows", 120, "uint32", "N*S*B"),
                PushConstantField("rope_n_dims", 124, "uint32", 128),
                PushConstantField("rope_freq_scale", 128, "float32", 1.0),
                PushConstantField("rope_freq_base", 132, "float32", 1_000_000.0),
                PushConstantField("rope_ext_factor", 136, "float32", 0.0),
                PushConstantField("rope_attn_factor", 140, "float32", 1.0),
                PushConstantField("rope_corr_dim0", 144, "float32", 128.0),
                PushConstantField("rope_corr_dim1", 148, "float32", 128.0),
                PushConstantField("rope_theta_scale", 152, "float32", 0.8058421877614819),
                PushConstantField("rope_has_ff", 156, "uint32", 0),
                PushConstantField("rope_section0", 160, "int32", 0),
                PushConstantField("rope_section1", 164, "int32", 0),
                PushConstantField("rope_section2", 168, "int32", 0),
                PushConstantField("rope_section3", 172, "int32", 0),
                PushConstantField("rope_is_imrope", 176, "uint32", 0),
                PushConstantField("rope_is_back", 180, "uint32", 0),
                PushConstantField("rope_set_rows_stride", 184, "uint32", 1024),
                PushConstantField("rope_ne00", 188, "uint32", "D"),
                PushConstantField("rope_ne01", 192, "uint32", "N"),
                PushConstantField("rope_ne02", 196, "uint32", "S"),
                PushConstantField("rope_nb01", 200, "uint32", "D"),
                PushConstantField("rope_nb02", 204, "uint32", "D*N"),
                PushConstantField("rope_nb03", 208, "uint32", "D*N*S"),
                PushConstantField("rope_dst_nb01", 212, "uint32", "D"),
                PushConstantField("rope_dst_nb02", 216, "uint32", "D*N"),
                PushConstantField("rope_dst_nb03", 220, "uint32", "D*N*T"),
            ),
        ),
    ),
    specialization_constants={0: 0, 1: 1},
    source=_SOURCE,
    include_dirs=("copied/agentorch_shader_source/llama_cpp_glsl",),
    compile_defines=(
        "A_TYPE=float",
        "B_TYPE=float",
        "D_TYPE=float",
        "FLOAT_TYPE=float",
        "FLOAT_TYPEV2=vec2",
        "RMS_NORM_ROPE_FUSION=1",
        "ROPE_D_TYPE=float16_t",
    ),
)
