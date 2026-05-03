"""Qwen3 V projection KV-cache write shader."""

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

SET_ROWS_F16_I64_TOKEN_MAJOR = ShaderVariant(
    name="set_rows_f16_i64",
    family="set_rows",
    contract=ShaderContract(
        name="set_rows_f16_i64",
        inputs={
            "x": TensorContract(dtype="float32", shape=(1, "B", "S", "W")),
            "row_indices": TensorContract(dtype="int64", shape=("S",)),
        },
        outputs={"output": TensorContract(dtype="float16", shape=(1, "B", "T", "W"))},
        bindings=(
            Binding("x", 0, BindingAccess.READ),
            Binding("row_indices", 1, BindingAccess.READ),
            Binding("output", 2, BindingAccess.WRITE),
        ),
        dispatch=("W", "S", "B"),
        push_constants=PushConstantBlock(
            size=116,
            fields=(
                PushConstantField("ne", 0, "uint32", "x.numel"),
                PushConstantField("src0_ne0", 4, "uint32", "x.dim3"),
                PushConstantField("src0_ne1", 8, "uint32", "x.dim2"),
                PushConstantField("src0_ne2", 12, "uint32", "x.dim1"),
                PushConstantField("src0_ne3", 16, "uint32", "x.dim0"),
                PushConstantField("src0_nb0", 20, "uint32", 1),
                PushConstantField("src0_nb1", 24, "uint32", "x.dim3"),
                PushConstantField("src0_nb2", 28, "uint32", "x.dim3*x.dim2"),
                PushConstantField("src0_nb3", 32, "uint32", "x.dim3*x.dim2*x.dim1"),
                PushConstantField("src1_ne0", 36, "uint32", "row_indices.dim0"),
                PushConstantField("src1_ne1", 40, "uint32", 1),
                PushConstantField("src1_ne2", 44, "uint32", 1),
                PushConstantField("src1_ne3", 48, "uint32", 1),
                PushConstantField("src1_nb0", 52, "uint32", 1),
                PushConstantField("src1_nb1", 56, "uint32", "row_indices.dim0"),
                PushConstantField("src1_nb2", 60, "uint32", "row_indices.dim0"),
                PushConstantField("src1_nb3", 64, "uint32", "row_indices.dim0"),
                PushConstantField("dst_ne0", 68, "uint32", "output.dim3"),
                PushConstantField("dst_ne1", 72, "uint32", "output.dim2"),
                PushConstantField("dst_ne2", 76, "uint32", "output.dim1"),
                PushConstantField("dst_ne3", 80, "uint32", "output.dim0"),
                PushConstantField("dst_nb0", 84, "uint32", 1),
                PushConstantField("dst_nb1", 88, "uint32", "output.dim3"),
                PushConstantField("dst_nb2", 92, "uint32", "output.dim3*output.dim2"),
                PushConstantField("dst_nb3", 96, "uint32", "output.dim3*output.dim2*output.dim1"),
                PushConstantField("padding", 100, "uint32", 0),
                PushConstantField("param1", 104, "float32", 0.0),
                PushConstantField("param2", 108, "float32", 0.0),
                PushConstantField("param3", 112, "int32", 0),
            ),
        ),
    ),
    source=copied_shader_variant_source(
        "set_rows_f16_i64_token_major.py",
        "SET_ROWS_F16_I64_TOKEN_MAJOR",
    ),
    include_dirs=("copied/agentorch_shader_source/llama_cpp_glsl",),
    compile_defines=(
        "SET_ROWS=1",
        "DATA_A_F16=1",
        "B_TYPE=uvec2",
        "B_SIZE=64",
        "D_TYPE=float",
        "FLOAT_TYPE=float",
    ),
)
