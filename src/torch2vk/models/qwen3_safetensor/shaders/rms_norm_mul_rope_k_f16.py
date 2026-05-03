"""Qwen3 K projection norm, RoPE, and KV-cache write shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_shader_variant_source
from torch2vk.shader import Binding, BindingAccess, ShaderContract, ShaderVariant, TensorContract

RMS_NORM_MUL_ROPE_K_F16 = ShaderVariant(
    name="rms_norm_mul_rope_f32_f16",
    family="rms_norm_mul_rope",
    contract=ShaderContract(
        name="rms_norm_mul_rope_f32_f16",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "S", "N", "D")),
            "weight": TensorContract(dtype="float32", shape=("D",)),
            "position_ids": TensorContract(dtype="int64", shape=("S",)),
            "row_indices": TensorContract(dtype="int64", shape=("S",)),
        },
        outputs={"output": TensorContract(dtype="float16", shape=("B", "T", "N", "D"))},
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
        "rms_norm_mul_rope_f32_f16.py",
        "RMS_NORM_MUL_ROPE_F32_F16",
    ),
)
