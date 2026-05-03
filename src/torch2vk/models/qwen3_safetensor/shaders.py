"""Initial Qwen3 shader contracts.

These variants define the Python/Vulkan boundary. Their GLSL source is a
placeholder until the Vulkan backend is filled in.
"""

from __future__ import annotations

from torch2vk.shader import Binding, BindingAccess, ShaderContract, ShaderVariant, TensorContract

EMBEDDING_LOOKUP_BF16_F32 = ShaderVariant(
    name="embedding_lookup_bf16_f32",
    family="embedding_lookup",
    contract=ShaderContract(
        name="embedding_lookup_bf16_f32",
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
    source="// TODO: implement Vulkan GLSL for embedding_lookup_bf16_f32\n",
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
    source="// TODO: implement Vulkan GLSL for rms_norm_f32\n",
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
    source="// TODO: implement Vulkan GLSL for linear_bf16_f32\n",
)

