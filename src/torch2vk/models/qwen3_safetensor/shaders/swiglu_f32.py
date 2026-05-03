"""Qwen3 SwiGLU shader."""

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
        push_constants=PushConstantBlock(
            size=64,
            fields=(
                PushConstantField("N", 0, "uint32", "output.numel"),
                PushConstantField("ne00", 4, "uint32", "gate.dim2"),
                PushConstantField("ne20", 8, "uint32", "output.dim2"),
                PushConstantField("mode", 12, "uint32", 2),
                PushConstantField("alpha", 16, "float32", 0.0),
                PushConstantField("limit", 20, "float32", 0.0),
                PushConstantField("nb01", 24, "uint32", "gate.dim2"),
                PushConstantField("nb02", 28, "uint32", "gate.dim2*gate.dim1"),
                PushConstantField("nb03", 32, "uint32", "gate.numel"),
                PushConstantField("ne01", 36, "uint32", "gate.dim1"),
                PushConstantField("ne02", 40, "uint32", "gate.dim0"),
                PushConstantField("nb11", 44, "uint32", "output.dim2"),
                PushConstantField("nb12", 48, "uint32", "output.dim2*output.dim1"),
                PushConstantField("nb13", 52, "uint32", "output.numel"),
                PushConstantField("ne11", 56, "uint32", "output.dim1"),
                PushConstantField("ne12", 60, "uint32", "output.dim0"),
            ),
        ),
    ),
    source=copied_shader_variant_source("swiglu_f32.py", "SWIGLU_F32"),
)
