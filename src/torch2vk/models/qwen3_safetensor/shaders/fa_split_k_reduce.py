"""Qwen3 flash attention split-k reduce shader."""

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

FA_SPLIT_K_REDUCE = ShaderVariant(
    name="fa_split_k_reduce",
    family="flash_attention_reduce",
    contract=ShaderContract(
        name="fa_split_k_reduce",
        inputs={
            "split_k_input": TensorContract(dtype="float32", shape=("B", "S", "A")),
            "sinks_placeholder": TensorContract(dtype="float32", shape=("B", "S", "Q")),
        },
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "Q"))},
        bindings=(
            Binding("split_k_input", 0, BindingAccess.READ),
            Binding("sinks_placeholder", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("Q", "S", "B"),
        push_constants=PushConstantBlock(size=24),
    ),
    source=copied_shader_variant_source("fa_split_k_reduce.py", "FA_SPLIT_K_REDUCE"),
)
