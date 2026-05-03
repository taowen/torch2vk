"""Qwen3 embedding lookup shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_shader_variant_source
from torch2vk.shader import Binding, BindingAccess, ShaderContract, ShaderVariant, TensorContract

EMBEDDING_LOOKUP_BF16_F32 = ShaderVariant(
    name="embedding_lookup_bf16_f32_sequence",
    family="embedding_lookup",
    contract=ShaderContract(
        name="embedding_lookup_bf16_f32_sequence",
        inputs={
            "input_ids": TensorContract(dtype="int32", shape=("B", "S")),
            "weight": TensorContract(dtype="bfloat16", shape=("V", "H")),
        },
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "H"))},
        bindings=(
            Binding("output", 0, BindingAccess.WRITE),
            Binding("input_ids", 1, BindingAccess.READ),
            Binding("weight", 2, BindingAccess.READ),
        ),
        dispatch=("H", "S", "B"),
    ),
    source=copied_shader_variant_source(
        "embedding_lookup_bf16_f32_sequence.py",
        "EMBEDDING_LOOKUP_BF16_F32_SEQUENCE",
    ),
)
