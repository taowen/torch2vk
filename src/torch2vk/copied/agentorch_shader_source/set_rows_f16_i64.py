"""Exact llama.cpp `set_rows_f16_i64` shader binding."""

from __future__ import annotations

from pathlib import Path


LLAMA_CPP_GLSL_DIR = Path(__file__).with_name("llama_cpp_glsl")

COPY_TO_QUANT_SOURCE = """
#version 450

#include "types.glsl"

#if defined(SET_ROWS) && QUANT_K == 1
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
const uint BLOCK_SIZE = 512;
#else
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
const uint BLOCK_SIZE = 32;
#endif

layout (binding = 0) readonly buffer S {float data_s[];};

#if defined(SET_ROWS)
#include "generic_binary_head.glsl"
layout (binding = 1) readonly buffer C {B_TYPE data_i[];};
layout (binding = 2) writeonly buffer Q {A_TYPE data_q[];};

#if B_SIZE == 64
#define DATA_I_SWIZZLE .x
#else
#define DATA_I_SWIZZLE
#endif

#else
#include "generic_unary_head.glsl"
layout (binding = 1) writeonly buffer Q {A_TYPE data_q[];};
#endif

#if defined(DATA_A_F32) || defined(DATA_A_F16)
void quantize(uint dst_idx, uint src_idx)
{
    data_q[dst_idx] = A_TYPE(data_s[src_idx]);
}
#endif

#if defined(DATA_A_BF16)
void quantize(uint dst_idx, uint src_idx)
{
    data_q[dst_idx] = A_TYPE(fp32_to_bf16(data_s[src_idx]));
}
#endif

#if defined(SET_ROWS)

void main() {
#ifdef NEEDS_INIT_IQ_SHMEM
    init_iq_shmem(gl_WorkGroupSize);
#endif

    const uint idx = ((gl_WorkGroupID.z * 262144 + gl_WorkGroupID.y * 512 + gl_WorkGroupID.x) * BLOCK_SIZE + gl_LocalInvocationID.x) * QUANT_K;

    if (idx >= p.ne) {
        return;
    }

    uint i00, i01, i02, i03;
    get_indices(idx, i00, i01, i02, i03);

    uint i12 = fastmod(i03, p.ne12);
    uint i11 = fastmod(i02, p.ne11);
    uint i10 = i01;

    uint i1 = data_i[src1_idx(i10, i11, i12, 0) + get_boffset()] DATA_I_SWIZZLE;

    uint src0_idx = src0_idx(i00, i01, i02, i03) + get_aoffset();
    uint dst_idx = dst_idx(i00 / QUANT_K, i1, i02, i03) + get_doffset();

    quantize(dst_idx, src0_idx);
}

#else

void main() {
#ifdef NEEDS_INIT_IQ_SHMEM
    init_iq_shmem(gl_WorkGroupSize);
#endif

    const uint idx = (gl_WorkGroupID.z * 262144 + gl_WorkGroupID.y * 512 + gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x) * QUANT_K;

    if (idx >= p.ne) {
        return;
    }

    uint dst_idx = dst_idx_quant(idx, QUANT_K);
    uint src_idx = get_aoffset() + src0_idx(idx);

    quantize(dst_idx, src_idx);
}

#endif
""".strip()
