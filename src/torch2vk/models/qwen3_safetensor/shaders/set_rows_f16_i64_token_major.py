"""Qwen3 V projection KV-cache write shader."""

from __future__ import annotations

from torch2vk.copied_shader_source import copied_shader_variant_source
from torch2vk.shader import Binding, BindingAccess, ShaderContract, ShaderVariant, TensorContract

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
    ),
    source=copied_shader_variant_source(
        "set_rows_f16_i64_token_major.py",
        "SET_ROWS_F16_I64_TOKEN_MAJOR",
    ),
)
