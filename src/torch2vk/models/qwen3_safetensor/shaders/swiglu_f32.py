"""Qwen3 SwiGLU shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_shader_variant_source
from torch2vk.shader import Binding, BindingAccess, ShaderContract, ShaderVariant, TensorContract

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
    ),
    source=copied_shader_variant_source("swiglu_f32.py", "SWIGLU_F32"),
)
