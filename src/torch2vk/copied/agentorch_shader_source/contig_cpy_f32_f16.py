"""Exact llama.cpp `contig_cpy_f32_f16` shader binding."""

from __future__ import annotations

from pathlib import Path

from agentorch.kernel.contract import (
    ceil_div,
    input_tensor,
    mul,
    output_tensor,
    shader_contract,
    storage_buffer_binding,
)
from agentorch.types import ANY_STRIDED_LAYOUT

from .llama_push_constants import unary_push_constant_block
from .shader_variant import shader_variant


_LLAMA_CPP_GLSL_DIR = Path(__file__).with_name("llama_cpp_glsl")


CONTIG_CPY_F32_F16 = shader_variant(
    name="contig_cpy_f32_f16",
    family="contig_cpy_f32_f16",
    contract=shader_contract(
        class_name="ContigCopyF32F16Program",
        shader_name="contig_cpy_f32_f16",
        fields=(
            input_tensor(
                name="x",
                binding="t_input",
                role="x",
                dtypes=("float32",),
                shape=("B", "M", "N", "K"),
                layout=ANY_STRIDED_LAYOUT,
            ),
            output_tensor(name="output", binding="t_output", role="output", dtypes=("float16",), shape=("B", "M", "N", "K")),
        ),
        uniforms=(),
        push_constants=unary_push_constant_block(src0_name="x", dst_name="output"),
        dispatch=(1, ceil_div(mul(mul("B", "M"), mul("N", "K")), 512), 1),
        bindings=(
            storage_buffer_binding(name="t_input", binding=0),
            storage_buffer_binding(name="t_output", binding=1),
        ),
    ),
    include_dirs=(_LLAMA_CPP_GLSL_DIR,),
    compile_defines=("A_TYPE=float", "D_TYPE=float16_t"),
    source="""
#version 450

#include "types.glsl"
#include "generic_unary_head.glsl"

#extension GL_EXT_control_flow_attributes : require

const uint num_threads = 128;

layout(local_size_x = num_threads, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint idx = get_idx();

    // num_threads * num_iter must equal 512, to match the wg_denoms and get_idx calculation
    const uint num_iter = 4;

    // fast path for when all four iterations are in-bounds
    if (idx + (num_iter-1)*num_threads < p.ne) {
        [[unroll]] for (uint i = 0; i < num_iter; ++i) {

#if defined(DATA_D_BF16)
            float f = float(data_a[get_aoffset() + idx]);
            data_d[get_doffset() + idx] = D_TYPE(fp32_to_bf16(f));
#elif !defined(OPTIMIZATION_ERROR_WORKAROUND)
            data_d[get_doffset() + idx] = D_TYPE(data_a[get_aoffset() + idx]);
#else
            data_d[get_doffset() + idx] = data_a[get_aoffset() + idx];
#endif
            idx += num_threads;
        }
    } else {
        [[unroll]] for (uint i = 0; i < num_iter; ++i) {
            if (idx >= p.ne) {
                continue;
            }

#if defined(DATA_D_BF16)
            float f = float(data_a[get_aoffset() + idx]);
            data_d[get_doffset() + idx] = D_TYPE(fp32_to_bf16(f));
#elif !defined(OPTIMIZATION_ERROR_WORKAROUND)
            data_d[get_doffset() + idx] = D_TYPE(data_a[get_aoffset() + idx]);
#else
            data_d[get_doffset() + idx] = data_a[get_aoffset() + idx];
#endif
            idx += num_threads;
        }
    }
}
""".strip(),
)
