"""Qwen3 RMS norm shader."""

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

_SOURCE = (
    copied_assignment_string(
        "rms_norm_f32_f32_weight_llama_wg512.py",
        "_RMS_NORM_COMP_SOURCE",
    )
    .replace(
        "layout(binding = 1) readonly buffer B { float data_b[]; };",
        "layout(binding = 1) readonly buffer B { uint data_b[]; };",
    )
    .replace(
        "float(data_b[b_offset + fastmod(col, p.ne10)])",
        "bf16_to_fp32(data_b[b_offset + fastmod(col, p.ne10)])",
    )
    .replace(
        "float(data_b[b_offset + col])",
        "bf16_to_fp32(data_b[b_offset + col])",
    )
)

RMS_NORM_F32 = ShaderVariant(
    name="rms_norm_f32_bf16_weight_llama_wg512",
    family="rms_norm",
    contract=ShaderContract(
        name="rms_norm_f32_bf16_weight_llama_wg512",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "S", "H")),
            "weight": TensorContract(dtype="bfloat16", shape=("H",)),
        },
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "H"))},
        bindings=(
            Binding("x", 0, BindingAccess.READ),
            Binding("weight", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("S", "B", 1),
        push_constants=PushConstantBlock(size=112),
    ),
    source=_SOURCE,
)
