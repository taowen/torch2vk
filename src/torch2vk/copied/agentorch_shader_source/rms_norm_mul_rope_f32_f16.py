"""Exact llama.cpp fused `rms_norm_mul_rope_f32_f16` shader binding."""

from __future__ import annotations

from pathlib import Path

from agentorch.kernel.contract import (
    input_tensor,
    output_tensor,
    shader_contract,
    storage_buffer_binding,
    tensor_binding_default,
)

from .llama_push_constants import rms_norm_mul_rope_push_constant_block
from .shader_variant import shader_variant


LLAMA_CPP_GLSL_DIR = Path(__file__).with_name("llama_cpp_glsl")

RMS_NORM_COMP_SOURCE = """
#version 450

#include "generic_binary_head.glsl"
#include "types.glsl"

#if RMS_NORM_ROPE_FUSION

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};

shared FLOAT_TYPE rope_data_a[1024];
#define data_d rope_data_a

layout (binding = 3) readonly buffer R_Y {int rope_data_pos[];};
layout (binding = 4) readonly buffer R_Z {float rope_data_ff[];};
layout (binding = 5) writeonly buffer R_D {ROPE_D_TYPE rope_data_d[];};
layout (binding = 6) readonly buffer R_I {uvec2 rope_data_i[];};

#include "rope_params.glsl"
#include "rope_funcs.glsl"

#define GGML_ROPE_TYPE_NORMAL 0
#define GGML_ROPE_TYPE_NEOX   2
#define GGML_ROPE_TYPE_MROPE  8
#define GGML_ROPE_TYPE_VISION 24

#endif

#extension GL_EXT_control_flow_attributes : enable
#define BLOCK_SIZE 512

layout (constant_id = 1) const bool do_multiply = false;

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

shared FLOAT_TYPE sumsh[BLOCK_SIZE];

void rms_norm(uint num_iters) {
    const uint ncols     = p.ne00;
    const uint nrows     = gl_NumWorkGroups.x;
    const uint nchannels = gl_NumWorkGroups.y;

    const uint row       = gl_WorkGroupID.x;
    const uint channel   = gl_WorkGroupID.y;
    const uint samp      = gl_WorkGroupID.z;
    const uint tid       = gl_LocalInvocationID.x;

    const uint stride_row       = p.nb01;
    const uint stride_channel   = p.nb02;
    const uint stride_sample    = p.nb03;

    uint32_t a_offset = samp*stride_sample + channel*stride_channel + row*stride_row + get_aoffset();
    uint32_t b_offset = src1_idx(0, row, channel, samp) + get_boffset();
#if RMS_NORM_ROPE_FUSION
    uint32_t d_offset = 0;
#else
    uint32_t d_offset = ((samp*nchannels + channel)*nrows + row)*ncols + get_doffset();
#endif
    FLOAT_TYPE sum = FLOAT_TYPE(0.0f);

    [[unroll]] for (uint col = tid, idx = 0; idx < num_iters; col += BLOCK_SIZE, ++idx) {
        FLOAT_TYPE xi = FLOAT_TYPE(0);
        if (col < ncols) {
            xi = FLOAT_TYPE(data_a[a_offset + col]);
        }
        sum += xi * xi;
    }

    sumsh[tid] = sum;
    barrier();
    [[unroll]] for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum += sumsh[tid + s];
            sumsh[tid] = sum;
        }
        barrier();
    }
    sum = sumsh[0];

    const FLOAT_TYPE mean = sum / FLOAT_TYPE(ncols);
    const FLOAT_TYPE scale = inversesqrt(mean + FLOAT_TYPE(p.param1));

    if (do_multiply) {
        if (ncols > p.ne10) {
            [[unroll]] for (uint col = tid, idx = 0; idx < num_iters; col += BLOCK_SIZE, ++idx) {
                if (col >= ncols) {
                    continue;
                }
                data_d[d_offset + col] = D_TYPE(scale * FLOAT_TYPE(data_a[a_offset + col]) * FLOAT_TYPE(data_b[b_offset + fastmod(col, p.ne10)]));
            }
        } else {
            [[unroll]] for (uint col = tid, idx = 0; idx < num_iters; col += BLOCK_SIZE, ++idx) {
                if (col >= ncols) {
                    continue;
                }
                data_d[d_offset + col] = D_TYPE(scale * FLOAT_TYPE(data_a[a_offset + col]) * FLOAT_TYPE(data_b[b_offset + col]));
            }
        }
    } else {
        [[unroll]] for (uint col = tid, idx = 0; idx < num_iters; col += BLOCK_SIZE, ++idx) {
            if (col >= ncols) {
                continue;
            }
            data_d[d_offset + col] = D_TYPE(scale * FLOAT_TYPE(data_a[a_offset + col]));
        }
    }
#if RMS_NORM_ROPE_FUSION
    barrier();
    rope_params rp = p.rope;
    for (uint t = 2*tid; t < ncols; t += 2*BLOCK_SIZE) {
        if (rp.rope_mode == GGML_ROPE_TYPE_NEOX) {
            rope_neox(t, row, channel, samp, rp);
        } else if (rp.rope_mode == GGML_ROPE_TYPE_NORMAL) {
            rope_norm(t, row, channel, samp, rp);
        }
    }
#endif
}

void main() {
    uint num_blocks = (p.ne00 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 32) {
        rms_norm(num_blocks);
    } else if (num_blocks > 16) {
        rms_norm(32);
    } else if (num_blocks > 12) {
        rms_norm(16);
    } else if (num_blocks > 10) {
        rms_norm(12);
    } else if (num_blocks > 8) {
        rms_norm(10);
    } else if (num_blocks > 4) {
        rms_norm(8);
    } else if (num_blocks == 4) {
        rms_norm(4);
    } else if (num_blocks == 3) {
        rms_norm(3);
    } else if (num_blocks == 2) {
        rms_norm(2);
    } else if (num_blocks == 1) {
        rms_norm(1);
    }
}
""".strip()


RMS_NORM_MUL_ROPE_F32_F16 = shader_variant(
    name="rms_norm_mul_rope_f32_f16",
    family="rms_norm_mul_rope_f32_f16",
    contract=shader_contract(
        class_name="RmsNormMulRopeF32F16Program",
        shader_name="rms_norm_mul_rope_f32_f16",
        fields=(
            input_tensor(
                name="x",
                binding="t_input",
                role="x",
                dtypes=("float32",),
                shape=("B", "S", "H", "D"),
            ),
            input_tensor(
                name="weight", binding="t_weight", role="weight", dtypes=("float32",), shape=("D",)
            ),
            input_tensor(
                name="mul_output_placeholder",
                binding="t_mul_output_placeholder",
                role="mul_output_placeholder",
                dtypes=("float32",),
                shape=("B", "S", "H", "D"),
            ),
            input_tensor(
                name="position_ids",
                binding="t_position_ids",
                role="position_ids",
                dtypes=("int32",),
                shape=("B", "S"),
            ),
            input_tensor(
                name="freq_factors_placeholder",
                binding="t_freq_factors_placeholder",
                role="freq_factors_placeholder",
                dtypes=("float32",),
                shape=("D",),
            ),
            output_tensor(
                name="output",
                binding="t_output",
                role="output",
                dtypes=("float16",),
                shape=("O0", "O1", "O2", "D"),
            ),
            input_tensor(
                name="row_indices",
                binding="t_row_indices",
                role="row_indices",
                dtypes=("int64",),
                shape=("S",),
            ),
        ),
        uniforms=(),
        push_constants=rms_norm_mul_rope_push_constant_block(
            src0_name="x",
            src1_name="weight",
            dst_name="output",
            n_dims=128,
            n_ctx_orig=40960,
            freq_base=1_000_000.0,
            rope_mode=2,
            set_rows_stride=1024,
            param1=1.0e-6,
        ),
        dispatch=("H", "S", "B"),
        tensor_defaults=(
            tensor_binding_default(
                field_name="mul_output_placeholder",
                source_field_name="x",
            ),
            tensor_binding_default(
                field_name="freq_factors_placeholder",
                source_field_name="weight",
            ),
        ),
        bindings=(
            storage_buffer_binding(name="t_input", binding=0),
            storage_buffer_binding(name="t_weight", binding=1),
            storage_buffer_binding(name="t_mul_output_placeholder", binding=2),
            storage_buffer_binding(name="t_position_ids", binding=3),
            storage_buffer_binding(name="t_freq_factors_placeholder", binding=4),
            storage_buffer_binding(name="t_output", binding=5),
            storage_buffer_binding(name="t_row_indices", binding=6),
        ),
    ),
    specialization_constants={0: 0, 1: 1},
    include_dirs=(LLAMA_CPP_GLSL_DIR,),
    compile_defines=(
        "A_TYPE=float",
        "B_TYPE=float",
        "D_TYPE=float",
        "FLOAT_TYPE=float",
        "FLOAT_TYPEV2=vec2",
        "RMS_NORM_ROPE_FUSION=1",
        "ROPE_D_TYPE=float16_t",
    ),
    source=RMS_NORM_COMP_SOURCE,
)
