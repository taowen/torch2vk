"""Exact llama.cpp `get_rows_f32_f32` shader binding."""

from __future__ import annotations

from agentorch.kernel.contract import ceil_div, input_tensor, output_tensor, shader_contract, storage_buffer_binding

from .llama_push_constants import binary_push_constant_block
from .shader_variant import shader_variant


_GET_ROWS_SOURCE = """
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





uint fastmod(uint a, uint b) {
    if ( (b & (b - 1)) == 0) {
        return a & (b - 1);
    }
    return a % b;
}

uint fastdiv(uint a, uint b) {
    return(a < b) ? 0 : (a / b);
}

void get_indices(uint idx, out uint i00, out uint i01, out uint i02, out uint i03, uint ne00, uint ne01, uint ne02, uint ne03) {
    i03 = fastdiv(idx, (ne02 * ne01 * ne00));
    const uint i03_offset = i03 * ne02 * ne01 * ne00;
    i02 = fastdiv( (idx - i03_offset), (ne01 * ne00));
    const uint i02_offset = i02 * ne01 * ne00;
    i01 = (idx - i03_offset - i02_offset) / ne00;
    i00 = idx - i03_offset - i02_offset - i01 * ne00;
}






layout(push_constant) uniform parameter
{
    uint ne;
    uint ne00; uint ne01; uint ne02; uint ne03; uint nb00; uint nb01; uint nb02; uint nb03;
    uint ne10; uint ne11; uint ne12; uint ne13; uint nb10; uint nb11; uint nb12; uint nb13;
    uint ne20; uint ne21; uint ne22; uint ne23; uint nb20; uint nb21; uint nb22; uint nb23;
    uint misalign_offsets;
    float param1; float param2; int param3;



} p;


layout(binding = 0) readonly buffer A { float data_a[]; };







layout(binding = 1) readonly buffer B { int data_b[]; };
layout(binding = 2) writeonly buffer D { float data_d[]; };



layout(constant_id = 0) const bool norepeat = false;

uint get_idx() {
    return gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
}

uint get_aoffset() { return p.misalign_offsets >> 16; }
uint get_boffset() { return(p.misalign_offsets >> 8) & 0xFF; }
uint get_doffset() { return p.misalign_offsets & 0xFF; }


void get_indices(uint idx, out uint i00, out uint i01, out uint i02, out uint i03) {
    get_indices(idx, i00, i01, i02, i03, p.ne00, p.ne01, p.ne02, p.ne03);
}

uint src0_idx(uint i00, uint i01, uint i02, uint i03) {
    return i03 * p.nb03 + i02 * p.nb02 + i01 * p.nb01 + i00 * p.nb00;
}

uint src1_idx(uint i00, uint i01, uint i02, uint i03) {
    if (norepeat) {
        return i03 * p.nb13 + i02 * p.nb12 + i01 * p.nb11 + i00 * p.nb10;
    } else {
        return fastmod(i03, p.ne13) * p.nb13 + fastmod(i02, p.ne12) * p.nb12 + fastmod(i01, p.ne11) * p.nb11 + fastmod(i00, p.ne10) * p.nb10;
    }
}

uint dst_idx(uint i00, uint i01, uint i02, uint i03) {
    return i03 * p.nb23 + i02 * p.nb22 + i01 * p.nb21 + i00 * p.nb20;
}

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint i00 = gl_GlobalInvocationID.x;

    if (i00 >= p.ne00) {
        return;
    }

    uint gid_z = gl_GlobalInvocationID.z;
    while (gid_z < p.ne11 * p.ne12) {
        uint gid_y = gl_GlobalInvocationID.y;
        while (gid_y < p.ne10) {
            const uint i10 = gid_y;
            const uint i11 = gid_z / p.ne12;
            const uint i12 = gid_z % p.ne12;

            const uint i01 = data_b[get_boffset() + i10 * p.nb10 + i11 * p.nb11 + i12 * p.nb12];

            const uint a_offset = get_aoffset() + i01 * p.nb01 + i11 * p.nb02 + i12 * p.nb03;
            const uint d_offset = get_doffset() + i10 * p.nb21 + i11 * p.nb22 + i12 * p.nb23;




                    float v = float(data_a[a_offset + i00]);


            data_d[d_offset + i00] = float(v);



            gid_y += gl_WorkGroupSize.y * gl_NumWorkGroups.y;
        }
        gid_z += gl_WorkGroupSize.z * gl_NumWorkGroups.z;
    }
}
"""


GET_ROWS_F32_F32 = shader_variant(
    name="get_rows_f32_f32",
    family="get_rows_f32_f32",
    contract=shader_contract(
        class_name="GetRowsF32F32Program",
        shader_name="get_rows_f32_f32",
        fields=(
            input_tensor(name="x", binding="t_x", role="x", dtypes=("float32",), shape=("B", "S", "H")),
            input_tensor(name="row_indices", binding="t_row_indices", role="row_indices", dtypes=("int32",), shape=("R",)),
            output_tensor(name="output", binding="t_output", role="output", dtypes=("float32",), shape=("B", "R", "H")),
        ),
        uniforms=(),
        push_constants=binary_push_constant_block(src0_name="x", src1_name="row_indices", dst_name="output", ne_name="x"),
        dispatch=(ceil_div("H", 512), "R", "B"),
        bindings=(
            storage_buffer_binding(name="t_x", binding=0),
            storage_buffer_binding(name="t_row_indices", binding=1),
            storage_buffer_binding(name="t_output", binding=2),
        ),
    ),
    source=_GET_ROWS_SOURCE,
)
