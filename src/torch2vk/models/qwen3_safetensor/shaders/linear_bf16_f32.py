"""Qwen3 BF16 linear shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_assignment_string
from torch2vk.shader import (
    Binding,
    BindingAccess,
    PushConstantBlock,
    ShaderContract,
    ShaderVariant,
    TensorContract,
)

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
            Binding("x", 0, BindingAccess.READ),
            Binding("weight", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("N", "S", "B"),
        push_constants=PushConstantBlock(size=68),
    ),
    source=copied_assignment_string(
        "matmul_bf16_f32_f16acc_aligned_l.py",
        "_MATMUL_BF16_F32_F16ACC_ALIGNED_CM1_SOURCE",
    ),
)
