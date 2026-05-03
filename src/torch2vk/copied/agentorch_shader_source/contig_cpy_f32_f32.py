"""Exact llama.cpp `contig_cpy_f32_f32` shader binding."""

from __future__ import annotations

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



CONTIG_CPY_F32_F32 = shader_variant(
    name="contig_cpy_f32_f32",
    family="contig_cpy_f32_f32",
    contract=shader_contract(
        class_name="ContigCopyF32F32Program",
        shader_name="contig_cpy_f32_f32",
        fields=(
            input_tensor(
                name="x",
                binding="t_input",
                role="x",
                dtypes=("float32",),
                shape=("B", "M", "N", "K"),
                layout=ANY_STRIDED_LAYOUT,
            ),
            output_tensor(
                name="output",
                binding="t_output",
                role="output",
                dtypes=("float32",),
                shape=("B", "M", "N", "K"),
            ),
        ),
        uniforms=(),
        push_constants=unary_push_constant_block(src0_name="x", dst_name="output"),
        dispatch=(1, ceil_div(mul(mul("B", "M"), mul("N", "K")), 512), 1),
        bindings=(
            storage_buffer_binding(name="t_input", binding=0),
            storage_buffer_binding(name="t_output", binding=1),
        ),
    ),
    source="""
#version 450





#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_16bit_storage : require











































struct block_q4_0
{
    float16_t d;
    uint8_t qs[16];
};
struct block_q4_0_packed16
{
    float16_t d;
    uint16_t qs[16 / 2];
};













struct block_q4_1
{
    float16_t d;
    float16_t m;
    uint8_t qs[16];
};

struct block_q4_1_packed16
{
    float16_t d;
    float16_t m;
    uint16_t qs[16 / 2];
};

struct block_q4_1_packed32
{
    f16vec2 dm;
    uint32_t qs[16 / 4];
};














struct block_q5_0
{
    float16_t d;
    uint16_t qh[2];
    uint8_t qs[16];
};

struct block_q5_0_packed16
{
    float16_t d;
    uint16_t qh[2];
    uint16_t qs[16 / 2];
};













struct block_q5_1
{
    float16_t d;
    float16_t m;
    uint qh;
    uint8_t qs[16];
};

struct block_q5_1_packed16
{
    float16_t d;
    float16_t m;
    uint qh;
    uint16_t qs[16 / 2];
};

struct block_q5_1_packed32
{
    f16vec2 dm;
    uint qh;
    uint32_t qs[16 / 4];
};














struct block_q8_0
{
    float16_t d;
    int8_t qs[32];
};

struct block_q8_0_packed16
{
    float16_t d;
    int16_t qs[32 / 2];
};













struct block_q1_0
{
    float16_t d;
    uint8_t qs[128 / 8];
};











struct block_q8_1
{
    f16vec2 ds;
    int8_t qs[32];
};

struct block_q8_1_packed16
{
    f16vec2 ds;
    int16_t qs[16];
};

struct block_q8_1_packed32
{
    f16vec2 ds;
    int32_t qs[8];
};


struct block_q8_1_x4
{
    f16vec2 ds[4];
    int32_t qs[32];
};

struct block_q8_1_x4_packed128
{
    f16vec2 ds[4];
    ivec4 qs[8];
};




struct block_q2_K
{
    uint8_t scales[256 / 16];
    uint8_t qs[256 / 4];
    f16vec2 dm;
};

struct block_q2_K_packed16
{
    uint16_t scales[256 / 16 / 2];
    uint16_t qs[256 / 4 / 2];
    f16vec2 dm;
};

struct block_q2_K_packed32
{
    uint32_t scales[256 / 16 / 4];
    uint32_t qs[256 / 4 / 4];
    f16vec2 dm;
};













struct block_q3_K
{
    uint8_t hmask[256 / 8];
    uint8_t qs[256 / 4];
    uint8_t scales[12];
    float16_t d;
};

struct block_q3_K_packed16
{
    uint16_t hmask[256 / 8 / 2];
    uint16_t qs[256 / 4 / 2];
    uint16_t scales[12 / 2];
    float16_t d;
};











struct block_q4_K
{
    f16vec2 dm;
    uint8_t scales[3 * 256 / 64];
    uint8_t qs[256 / 2];
};

struct block_q4_K_packed16
{
    f16vec2 dm;
    uint16_t scales[3 * 256 / 64 / 2];
    uint16_t qs[256 / 2 / 2];
};

struct block_q4_K_packed32
{
    f16vec2 dm;
    uint32_t scales[3 * 256 / 64 / 4];
    uint32_t qs[256 / 2 / 4];
};

struct block_q4_K_packed128
{
    uvec4 q4k[9];
};












struct block_q5_K
{
    f16vec2 dm;
    uint8_t scales[12];
    uint8_t qh[256 / 8];
    uint8_t qs[256 / 2];
};

struct block_q5_K_packed16
{
    f16vec2 dm;
    uint16_t scales[12 / 2];
    uint16_t qh[256 / 8 / 2];
    uint16_t qs[256 / 2 / 2];
};

struct block_q5_K_packed32
{
    f16vec2 dm;
    uint32_t scales[12 / 4];
    uint32_t qh[256 / 8 / 4];
    uint32_t qs[256 / 2 / 4];
};

struct block_q5_K_packed128
{
    uvec4 q5k[11];
};












struct block_q6_K
{
    uint8_t ql[256 / 2];
    uint8_t qh[256 / 4];
    int8_t scales[256 / 16];
    float16_t d;
};

struct block_q6_K_packed16
{
    uint16_t ql[256 / 2 / 2];
    uint16_t qh[256 / 4 / 2];
    int16_t scales[256 / 16 / 2];
    float16_t d;
};














struct block_iq1_s {
    float16_t d;
    uint8_t qs[256 / 8];
    uint16_t qh[256 / 32];
};

struct block_iq1_s_packed16 {
    float16_t d;
    uint16_t qs[256 / 8 / 2];
    uint16_t qh[256 / 32];
};




struct block_iq1_m {
    uint8_t qs[256 / 8];
    uint8_t qh[256 / 16];
    uint16_t scales[256 / 64];
};

struct block_iq1_m_packed16 {
    uint16_t qs[256 / 8 / 2];
    uint16_t qh[256 / 16 / 2];
    uint16_t scales[256 / 64];
};

struct block_iq1_m_packed32 {
    uint32_t qs[256 / 8 / 4];
    uint32_t qh[256 / 16 / 4];
    uint32_t scales[256 / 64 / 2];
};

struct block_iq1_m_packed64 {
    uint64_t qs[256 / 8 / 8];
    uint64_t qh[256 / 16 / 8];
    uint64_t scales;
};


























































































































































































































































































































































































































































struct block_iq2_xxs
{
    float16_t d;
    uint8_t qs[256 / 4];
};

struct block_iq2_xxs_packed16
{
    float16_t d;
    uint16_t qs[256 / 8];
};





























































































struct block_iq2_xs
{
    float16_t d;
    uint16_t qs[256 / 8];
    uint8_t scales[256 / 32];
};

struct block_iq2_xs_packed16
{
    float16_t d;
    uint16_t qs[256 / 8];
    uint16_t scales[256 / 64];
};





























































































































































struct block_iq2_s
{
    float16_t d;
    uint8_t qs[256 / 4];
    uint8_t qh[256 / 32];
    uint8_t scales[256 / 32];
};

struct block_iq2_s_packed16
{
    float16_t d;
    uint16_t qs[256 / 8];
    uint16_t qh[256 / 64];
    uint16_t scales[256 / 64];
};





























































































































































































































































































struct block_iq3_xxs
{
    float16_t d;
    uint8_t qs[256 / 4 + 256 / 8];
};

struct block_iq3_xxs_packed16
{
    float16_t d;
    uint16_t qs[256 / 8 + 256 / 16];
};





























































struct block_iq3_s
{
    float16_t d;
    uint8_t qs[256 / 4];
    uint8_t qh[256 / 32];
    uint8_t signs[256 / 8];
    uint8_t scales[256 / 64];
};

struct block_iq3_s_packed16
{
    float16_t d;
    uint16_t qs[256 / 4 / 2];
    uint16_t qh[256 / 32 / 2];
    uint16_t signs[256 / 8 / 2];
    uint16_t scales[256 / 64 / 2];
};





























































































struct block_iq4_xs
{
    float16_t d;
    uint16_t scales_h;
    uint8_t scales_l[256 / 64];
    uint8_t qs[256 / 2];
};

struct block_iq4_xs_packed16
{
    float16_t d;
    uint16_t scales_h;
    uint16_t scales_l[256 / 128];
    uint16_t qs[256 / 4];
};

struct block_iq4_xs_packed32
{
    float16_t d;
    uint16_t scales_h;
    uint32_t scales_l;
    uint32_t qs[256 / 8];
};












struct block_iq4_nl
{
    float16_t d;
    uint8_t qs[32 / 2];
};

struct block_iq4_nl_packed16
{
    float16_t d;
    uint16_t qs[32 / 2 / 2];
};












struct block_mxfp4
{
    uint8_t e;
    uint8_t qs[32 / 2];
};











struct block_nvfp4
{
    uint8_t d[64 / 16];
    uint8_t qs[64 / 2];
};







































































uint32_t fp32_to_bf16(float f)
{
    uint32_t u = floatBitsToUint(f);
    u = (u + (0x7fff + ( (u >> 16) & 1))) >> 16;
    return u;
}

float bf16_to_fp32(uint32_t u)
{
    return uintBitsToFloat(u << 16);
}

vec4 bf16_to_fp32(uvec4 u)
{
    return vec4(bf16_to_fp32(u.x), bf16_to_fp32(u.y), bf16_to_fp32(u.z), bf16_to_fp32(u.w));
}

float e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = x;
        bits = bits << 23;
    }

    return uintBitsToFloat(bits);
}























#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require

layout(push_constant) uniform parameter
{
    uint ne;
    uint ne00; uint ne01; uint ne02; uint ne03; uint nb00; uint nb01; uint nb02; uint nb03;
    uint ne10; uint ne11; uint ne12; uint ne13; uint nb10; uint nb11; uint nb12; uint nb13;
    uint misalign_offsets;
    float param1; float param2;

    uint ne0_012mp; uint ne0_012L;
    uint ne0_01mp; uint ne0_01L;
    uint ne0_0mp; uint ne0_0L;
    uint ne1_012mp; uint ne1_012L;
    uint ne1_01mp; uint ne1_01L;
    uint ne1_0mp; uint ne1_0L;
} p;

layout(binding = 0) readonly buffer A { float data_a[]; };







layout(binding = 1) writeonly buffer D { float data_d[]; };

uint get_idx() {
    return gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
}

uint get_aoffset() { return p.misalign_offsets >> 16; }
uint get_doffset() { return p.misalign_offsets & 0xFFFF; }


uint fastdiv(uint n, uint mp, uint L) {
    uint msbs, lsbs;

    umulExtended(n, mp, msbs, lsbs);
    return(msbs + n) >> L;
}

uint src0_idx(uint idx) {
    const uint i03 = fastdiv(idx, p.ne0_012mp, p.ne0_012L);
    const uint i03_offset = i03 * p.ne02 * p.ne01 * p.ne00;
    const uint i02 = fastdiv(idx - i03_offset, p.ne0_01mp, p.ne0_01L);
    const uint i02_offset = i02 * p.ne01 * p.ne00;
    const uint i01 = fastdiv(idx - i03_offset - i02_offset, p.ne0_0mp, p.ne0_0L);
    const uint i00 = idx - i03_offset - i02_offset - i01 * p.ne00;
    return i03 * p.nb03 + i02 * p.nb02 + i01 * p.nb01 + i00 * p.nb00;
}

uint dst_idx(uint idx) {
    const uint i13 = fastdiv(idx, p.ne1_012mp, p.ne1_012L);
    const uint i13_offset = i13 * p.ne12 * p.ne11 * p.ne10;
    const uint i12 = fastdiv(idx - i13_offset, p.ne1_01mp, p.ne1_01L);
    const uint i12_offset = i12 * p.ne11 * p.ne10;
    const uint i11 = fastdiv(idx - i13_offset - i12_offset, p.ne1_0mp, p.ne1_0L);
    const uint i10 = idx - i13_offset - i12_offset - i11 * p.ne10;
    return i13 * p.nb13 + i12 * p.nb12 + i11 * p.nb11 + i10 * p.nb10;
}

uint src0_idx_quant(uint idx, uint qk) {
    const uint i03 = fastdiv(idx, p.ne0_012mp, p.ne0_012L);
    const uint i03_offset = i03 * p.ne02 * p.ne01 * p.ne00;
    const uint i02 = fastdiv(idx - i03_offset, p.ne0_01mp, p.ne0_01L);
    const uint i02_offset = i02 * p.ne01 * p.ne00;
    const uint i01 = fastdiv(idx - i03_offset - i02_offset, p.ne0_0mp, p.ne0_0L);
    const uint i00 = idx - i03_offset - i02_offset - i01 * p.ne00;
    return i03 * p.nb03 + i02 * p.nb02 + i01 * p.nb01 + (i00 / qk) * p.nb00;
}

uint dst_idx_quant(uint idx, uint qk) {
    const uint i13 = fastdiv(idx, p.ne1_012mp, p.ne1_012L);
    const uint i13_offset = i13 * p.ne12 * p.ne11 * p.ne10;
    const uint i12 = fastdiv(idx - i13_offset, p.ne1_01mp, p.ne1_01L);
    const uint i12_offset = i12 * p.ne11 * p.ne10;
    const uint i11 = fastdiv(idx - i13_offset - i12_offset, p.ne1_0mp, p.ne1_0L);
    const uint i10 = idx - i13_offset - i12_offset - i11 * p.ne10;
    return i13 * p.nb13 + i12 * p.nb12 + i11 * p.nb11 + (i10 / qk) * p.nb10;
}

#extension GL_EXT_control_flow_attributes : require

const uint num_threads = 128;

layout(local_size_x = num_threads, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint idx = get_idx();


    const uint num_iter = 4;


    if (idx + (num_iter - 1) * num_threads < p.ne) {
        [[unroll]] for (uint i = 0; i < num_iter; ++ i) {





            data_d[get_doffset() + idx] = float(data_a[get_aoffset() + idx]);



            idx += num_threads;
        }
    } else {
        [[unroll]] for (uint i = 0; i < num_iter; ++ i) {
            if (idx >= p.ne) {
                continue;
            }





            data_d[get_doffset() + idx] = float(data_a[get_aoffset() + idx]);



            idx += num_threads;
        }
    }
}
""".strip(),
)
