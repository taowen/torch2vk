"""Qwen3 argmax stage 2 shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_shader_variant_source
from torch2vk.shader import Binding, BindingAccess, ShaderContract, ShaderVariant, TensorContract

ARGMAX_LAST_LOGITS_STAGE2 = ShaderVariant(
    name="argmax_last_logits_f32_stage2",
    family="argmax",
    contract=ShaderContract(
        name="argmax_last_logits_f32_stage2",
        inputs={
            "partial_values": TensorContract(dtype="float32", shape=("B", "C")),
            "partial_indices": TensorContract(dtype="int32", shape=("B", "C")),
        },
        outputs={"output": TensorContract(dtype="int32", shape=("B",))},
        bindings=(
            Binding("partial_values", 0, BindingAccess.READ),
            Binding("partial_indices", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("B", 1, 1),
    ),
    source=copied_shader_variant_source(
        "argmax_last_logits_f32_parallel.py",
        "ARGMAX_LAST_LOGITS_F32_STAGE2",
    ),
)
