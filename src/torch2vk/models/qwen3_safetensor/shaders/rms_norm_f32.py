"""Qwen3 RMS norm shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_assignment_string
from torch2vk.shader import Binding, BindingAccess, ShaderContract, ShaderVariant, TensorContract

RMS_NORM_F32 = ShaderVariant(
    name="rms_norm_f32",
    family="rms_norm",
    contract=ShaderContract(
        name="rms_norm_f32",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "S", "H")),
            "weight": TensorContract(dtype="float32", shape=("H",)),
        },
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "H"))},
        bindings=(
            Binding("x", 0, BindingAccess.READ),
            Binding("weight", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("S", "B", 1),
    ),
    source=copied_assignment_string(
        "rms_norm_f32_f32_weight_llama_wg512.py",
        "_RMS_NORM_COMP_SOURCE",
    ),
)
