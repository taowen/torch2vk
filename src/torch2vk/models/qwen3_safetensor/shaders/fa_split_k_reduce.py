"""Qwen3 flash attention split-k reduce shader."""

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

FA_SPLIT_K_REDUCE = ShaderVariant(
    name="fa_split_k_reduce",
    family="flash_attention_reduce",
    contract=ShaderContract(
        name="fa_split_k_reduce",
        inputs={
            "split_k_input": TensorContract(dtype="float32", shape=("SK",)),
            "sinks_placeholder": TensorContract(dtype="float32", shape=("B", "S", "QH", "D")),
        },
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "QH", "D"))},
        bindings=(
            Binding("split_k_input", 0, BindingAccess.READ),
            Binding("sinks_placeholder", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("QH", "D", "B*S"),
        push_constants=PushConstantBlock(
            size=24,
            fields=(
                PushConstantField("D", 0, "uint32", "D"),
                PushConstantField("ne1", 4, "uint32", "QH"),
                PushConstantField("ne2", 8, "uint32", "S"),
                PushConstantField("ne3", 12, "uint32", "B"),
                PushConstantField("k_num", 16, "uint32", 4),
                PushConstantField("sinks", 20, "uint32", 0),
            ),
        ),
    ),
    specialization_constants={0: 64},
    source=copied_shader_variant_source("fa_split_k_reduce.py", "FA_SPLIT_K_REDUCE"),
)
