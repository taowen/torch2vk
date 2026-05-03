"""Qwen3 Q projection norm and RoPE shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_shader_variant_source
from torch2vk.shader import Binding, BindingAccess, ShaderContract, ShaderVariant, TensorContract

RMS_NORM_MUL_ROPE_Q_F32 = ShaderVariant(
    name="rms_norm_mul_rope_f32_f32",
    family="rms_norm_mul_rope",
    contract=ShaderContract(
        name="rms_norm_mul_rope_f32_f32",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "S", "N", "D")),
            "weight": TensorContract(dtype="float32", shape=("D",)),
            "position_ids": TensorContract(dtype="int64", shape=("S",)),
            "row_indices": TensorContract(dtype="int64", shape=("S",)),
        },
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "N", "D"))},
        bindings=(
            Binding("x", 0, BindingAccess.READ),
            Binding("weight", 1, BindingAccess.READ),
            Binding("position_ids", 2, BindingAccess.READ),
            Binding("row_indices", 3, BindingAccess.READ),
            Binding("output", 4, BindingAccess.WRITE),
        ),
        dispatch=("D", "S", "B"),
    ),
    source=copied_shader_variant_source(
        "rms_norm_mul_rope_f32_f32.py",
        "RMS_NORM_MUL_ROPE_F32_F32",
    ),
)
