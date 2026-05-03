"""Qwen3 BF16 linear shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_assignment_string
from torch2vk.shader import (
    Binding,
    BindingAccess,
    PushConstantBlock,
    PushConstantField,
    ResourceBinding,
    ShaderContract,
    ShaderVariant,
    TensorContract,
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
        outputs={"output": TensorContract(dtype="float32", shape=("B", "S", "N"))},
        bindings=(
            Binding("weight", 0, BindingAccess.READ),
            Binding("x", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        resources=(
            ResourceBinding("fuse0_placeholder", 3, "storage_buffer"),
            ResourceBinding("fuse1_placeholder", 4, "storage_buffer"),
        ),
        dispatch=("N", "S", "B"),
        push_constants=PushConstantBlock(
            size=52,
            fields=(
                PushConstantField("ncols", 0, "uint32", "K"),
                PushConstantField("stride_a", 4, "uint32", "K"),
                PushConstantField("stride_b", 8, "uint32", "K"),
                PushConstantField("stride_d", 12, "uint32", "N"),
                PushConstantField("batch_stride_a", 16, "uint32", "N*K"),
                PushConstantField("batch_stride_b", 20, "uint32", "S*K"),
                PushConstantField("batch_stride_d", 24, "uint32", "S*N"),
                PushConstantField("fusion_flags", 28, "uint32", 0),
                PushConstantField("base_work_group_y", 32, "uint32", 0),
                PushConstantField("ne02", 36, "uint32", 1),
                PushConstantField("ne12", 40, "uint32", "B"),
                PushConstantField("broadcast2", 44, "uint32", 1),
                PushConstantField("broadcast3", 48, "uint32", 1),
            ),
        ),
    ),
    source=copied_assignment_string(
        "mul_mat_vec_f16_f32_f32.py",
        "_MUL_MAT_VEC_BF16_TORCH_PARITY_SOURCE",
    ),
)
