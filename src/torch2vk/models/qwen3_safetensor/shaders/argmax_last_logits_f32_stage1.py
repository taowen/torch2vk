"""Qwen3 argmax stage 1 shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_shader_variant_source
from torch2vk.shader import Binding, BindingAccess, ShaderContract, ShaderVariant, TensorContract

ARGMAX_LAST_LOGITS_STAGE1 = ShaderVariant(
    name="argmax_last_logits_f32_stage1",
    family="argmax",
    contract=ShaderContract(
        name="argmax_last_logits_f32_stage1",
        inputs={"logits": TensorContract(dtype="float32", shape=("B", "S", "V"))},
        outputs={
            "partial_values": TensorContract(dtype="float32", shape=("B", "C")),
            "partial_indices": TensorContract(dtype="int32", shape=("B", "C")),
        },
        bindings=(
            Binding("logits", 0, BindingAccess.READ),
            Binding("partial_values", 1, BindingAccess.WRITE),
            Binding("partial_indices", 2, BindingAccess.WRITE),
        ),
        dispatch=("C", "B", 1),
    ),
    source=copied_shader_variant_source(
        "argmax_last_logits_f32_parallel.py",
        "ARGMAX_LAST_LOGITS_F32_STAGE1",
    ),
)
