"""Exact llama.cpp `set_rows_f16_i64` bound to flattened GGML KV-cache views."""

from __future__ import annotations

from agentorch.kernel.contract import ceil_div, input_tensor, mul, output_tensor, shader_contract, storage_buffer_binding

from .llama_push_constants import binary_push_constant_block
from .set_rows_f16_i64 import COPY_TO_QUANT_SOURCE, LLAMA_CPP_GLSL_DIR
from .shader_variant import shader_variant


SET_ROWS_F16_I64_TOKEN_MAJOR = shader_variant(
    name="set_rows_f16_i64",
    family="set_rows_f16_i64",
    contract=shader_contract(
        class_name="SetRowsF16I64TokenMajorProgram",
        shader_name="set_rows_f16_i64",
        fields=(
            input_tensor(name="x", binding="t_x", role="x", dtypes=("float32",), shape=(1, "B", "S", "W")),
            input_tensor(name="row_indices", binding="t_row_indices", role="row_indices", dtypes=("int64",), shape=("S",)),
            output_tensor(name="output", binding="t_output", role="output", dtypes=("float16",), shape=(1, "B", "T", "W")),
        ),
        uniforms=(),
        push_constants=binary_push_constant_block(
            ne_name="x",
            src0_name="x",
            src1_name="row_indices",
            dst_name="output",
        ),
        dispatch=(ceil_div(mul(mul("B", "S"), "W"), 512), 1, 1),
        bindings=(
            storage_buffer_binding(name="t_x", binding=0),
            storage_buffer_binding(name="t_row_indices", binding=1),
            storage_buffer_binding(name="t_output", binding=2),
        ),
    ),
    compile_defines=(
        "SET_ROWS=1",
        "DATA_A_F16=1",
        "B_TYPE=uvec2",
        "B_SIZE=64",
        "D_TYPE=float",
        "FLOAT_TYPE=float",
    ),
    include_dirs=(LLAMA_CPP_GLSL_DIR,),
    source=COPY_TO_QUANT_SOURCE,
)
