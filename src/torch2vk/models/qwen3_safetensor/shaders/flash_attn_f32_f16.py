"""Qwen3 flash attention shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_shader_variant_source
from torch2vk.shader import (
    Binding,
    BindingAccess,
    PushConstantBlock,
    ShaderContract,
    ShaderVariant,
    TensorContract,
)

FLASH_ATTN_F32_F16 = ShaderVariant(
    name="flash_attn_f32_f16_aligned_f32accf16",
    family="flash_attention",
    contract=ShaderContract(
        name="flash_attn_f32_f16_aligned_f32accf16",
        inputs={
            "q": TensorContract(dtype="float32", shape=("B", "S", "QH", "D")),
            "k": TensorContract(dtype="float16", shape=("B", "T", "KH", "D")),
            "v": TensorContract(dtype="float16", shape=("B", "T", "KH", "D")),
            "mask": TensorContract(dtype="float16", shape=("B", 1, "S", "T")),
            "sinks_placeholder": TensorContract(dtype="float32", shape=("B", "S", "QH", "D")),
            "mask_opt_placeholder": TensorContract(dtype="float32", shape=("B", "S", "QH", "D")),
        },
        outputs={"split_k_output": TensorContract(dtype="float32", shape=("B", "S", "A"))},
        bindings=(
            Binding("q", 0, BindingAccess.READ),
            Binding("k", 1, BindingAccess.READ),
            Binding("v", 2, BindingAccess.READ),
            Binding("mask", 3, BindingAccess.READ),
            Binding("sinks_placeholder", 4, BindingAccess.READ),
            Binding("split_k_output", 5, BindingAccess.WRITE),
            Binding("mask_opt_placeholder", 6, BindingAccess.READ),
        ),
        dispatch=("S", "QH", "B"),
        push_constants=PushConstantBlock(size=128),
    ),
    source=copied_shader_variant_source(
        "flash_attn_f32_f16_aligned_f32accf16.py",
        "FLASH_ATTN_F32_F16_ALIGNED_F32ACCF16",
    ),
)
