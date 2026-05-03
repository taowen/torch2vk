"""Qwen3 residual add shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_shader_variant_source
from torch2vk.shader import Binding, BindingAccess, ShaderContract, ShaderVariant, TensorContract

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
    ),
    source=copied_shader_variant_source(
        "add_f32_f32_f32_norepeat.py",
        "ADD_F32_F32_F32_NOREPEAT",
    ),
)
