"""Exact llama.cpp rms_norm_mul_partials_f32 shader binding."""

from __future__ import annotations

from collections.abc import Mapping
from agentorch.types import DeviceTensorLike as DeviceTensor
from agentorch.kernel.contract import ceil_div, input_tensor, output_tensor, shader_contract, storage_buffer_binding

from .llama_push_constants import binary_push_constant_block
from .shader_variant import shader_variant


_RMS_NORM_PARTIALS_COMP_SOURCE = """
#version 450


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







layout(binding = 1) readonly buffer B { float data_b[]; };
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
























#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable



layout(constant_id = 1) const bool do_multiply = false;

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(binding = 3, std430) readonly buffer PartialsBuf { float partial_sums[]; };

shared float sumsh[128];

void main() {
    const uint ncols = p.ne00;
    const uint nrows = gl_NumWorkGroups.x;
    const uint nchannels = gl_NumWorkGroups.y;

    const uint row = 0;
    const uint channel = gl_WorkGroupID.y;
    const uint samp = gl_WorkGroupID.z;


    const uint tid = gl_GlobalInvocationID.x;

    const uint stride_row = p.nb01;
    const uint stride_channel = p.nb02;
    const uint stride_sample = p.nb03;

    uint32_t a_offset = samp * stride_sample + channel * stride_channel + row * stride_row + get_aoffset();
    uint32_t b_offset = src1_idx(0, row, channel, samp) + get_boffset();
    uint32_t d_offset = ( (samp * nchannels + channel) * nrows + row) * ncols + get_doffset();

             float sum = float(0.0f);

    uint32_t num_partials = p.param3;
    for (uint32_t i = gl_SubgroupInvocationID; i < num_partials; i += gl_SubgroupSize) {
        sum += partial_sums[i];
    }
    sum = subgroupAdd(sum);

    uint col = tid;
    if (col >= ncols) {
        return;
    }

    const float mean = sum / float(ncols);
    const float scale = inversesqrt(mean + float(p.param1));

    if (do_multiply) {
        if (ncols > p.ne10) {
            data_d[d_offset + col] = float(scale * float(data_a[a_offset + col]) * float(data_b[b_offset + fastmod(col, p.ne10)]));
        } else {
            data_d[d_offset + col] = float(scale * float(data_a[a_offset + col]) * float(data_b[b_offset + col]));
        }
    } else {
        data_d[d_offset + col] = float(scale * float(data_a[a_offset + col]));
    }
}
"""


RMS_NORM_MUL_PARTIALS_F32 = shader_variant(
    name="rms_norm_mul_partials_f32",
    family="rms_norm_mul_partials_f32",
    contract=shader_contract(
        class_name="RmsNormMulPartialsF32Program",
        shader_name="rms_norm_mul_partials_f32",
        fields=(
            input_tensor(name="x", binding="t_input", role="x", dtypes=("float32",), shape=("B", "S", "H")),
            input_tensor(name="weight", binding="t_weight_raw", role="weight", dtypes=("float32",), shape=("H",)),
            output_tensor(name="output", binding="t_output", role="output", dtypes=("float32",), shape=("B", "S", "H")),
            input_tensor(
                name="partial_sums",
                binding="t_partial_sums",
                role="partial_sums",
                dtypes=("float32",),
                shape=("B", "S", ceil_div("H", 512)),
            ),
        ),
        uniforms=(),
        push_constants=binary_push_constant_block(
            src0_name="x",
            src1_name="weight",
            dst_name="output",
            param1=1.0e-6,
            param3=lambda tensors, _shape: _partial_sums_count(tensors),
        ),
        dispatch=(ceil_div("H", 128), "S", "B"),
        bindings=(
            storage_buffer_binding(name="t_input", binding=0),
            storage_buffer_binding(name="t_weight_raw", binding=1),
            storage_buffer_binding(name="t_output", binding=2),
            storage_buffer_binding(name="t_partial_sums", binding=3),
        ),
    ),
    specialization_constants={0: 0, 1: 1},
    source=_RMS_NORM_PARTIALS_COMP_SOURCE,
)


def _partial_sums_count(tensors: Mapping[str, DeviceTensor]) -> int:
    return tensors["partial_sums"].shape[2]
