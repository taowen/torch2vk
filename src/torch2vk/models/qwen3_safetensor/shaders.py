"""Initial Qwen3 shader contracts backed by copied Agentorch GLSL."""

from __future__ import annotations

from torch2vk.copied_shader_source import (
    copied_assignment_string,
    copied_shader_variant_source,
)
from torch2vk.shader import (
    Binding,
    BindingAccess,
    ShaderContract,
    ShaderVariant,
    TensorContract,
)

EMBEDDING_LOOKUP_BF16_F32 = ShaderVariant(
    name="embedding_lookup_bf16_f32_sequence",
    family="embedding_lookup",
    contract=ShaderContract(
        name="embedding_lookup_bf16_f32_sequence",
        inputs={
            "input_ids": TensorContract(dtype="int32", shape=("B", "S")),
            "weight": TensorContract(dtype="bfloat16", shape=("V", "H")),
        },
        outputs={
            "output": TensorContract(dtype="float32", shape=("B", "S", "H")),
        },
        bindings=(
            Binding("input_ids", 0, BindingAccess.READ),
            Binding("weight", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("H", "S", "B"),
    ),
    source=copied_shader_variant_source(
        "embedding_lookup_bf16_f32_sequence.py",
        "EMBEDDING_LOOKUP_BF16_F32_SEQUENCE",
    ),
)


RMS_NORM_F32 = ShaderVariant(
    name="rms_norm_f32",
    family="rms_norm",
    contract=ShaderContract(
        name="rms_norm_f32",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "S", "H")),
            "weight": TensorContract(dtype="float32", shape=("H",)),
        },
        outputs={
            "output": TensorContract(dtype="float32", shape=("B", "S", "H")),
        },
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


LINEAR_BF16_F32 = ShaderVariant(
    name="linear_bf16_f32",
    family="linear",
    contract=ShaderContract(
        name="linear_bf16_f32",
        inputs={
            "x": TensorContract(dtype="float32", shape=("B", "S", "K")),
            "weight": TensorContract(dtype="bfloat16", shape=("N", "K")),
        },
        outputs={
            "output": TensorContract(dtype="float32", shape=("B", "S", "N")),
        },
        bindings=(
            Binding("x", 0, BindingAccess.READ),
            Binding("weight", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("N", "S", "B"),
    ),
    source=copied_assignment_string(
        "matmul_bf16_f32_f16acc_aligned_l.py",
        "_MATMUL_BF16_F32_F16ACC_ALIGNED_CM1_SOURCE",
    ),
)
