"""llama.cpp `add_rms_f32_f32_f32_norepeat` shader binding."""

from __future__ import annotations

from agentorch.kernel.contract import ceil_div, input_tensor, mul, output_tensor, shader_contract, storage_buffer_binding

from .llama_push_constants import binary_push_constant_block
from .shader_variant import shader_variant


_ADD_SOURCE = """
#version 450


#extension GL_EXT_shader_16bit_storage : require

#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable





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

const uint num_threads = 256;

layout(binding = 3, std430) buffer PartialBuf { float partial_sums[]; };

layout(local_size_x = num_threads, local_size_y = 1, local_size_z = 1) in;



shared float sumsh[num_threads];


void main() {
    uint idx = get_idx();
    uint orig_idx = idx;


    const uint num_iter = 2;

             float sum_sq = 0;

    [[unroll]] for (uint i = 0; i < num_iter; ++ i) {
        if (idx >= p.ne) {
            continue;
        }
        uint i00, i01, i02, i03;
        get_indices(idx, i00, i01, i02, i03);

                 float sum = float(data_a[get_aoffset() + src0_idx(i00, i01, i02, i03)]) + float(data_b[get_boffset() + src1_idx(i00, i01, i02, i03)]);
        sum_sq += sum * sum;

        data_d[get_doffset() + dst_idx(i00, i01, i02, i03)] = float(sum);

        idx += num_threads;
    }


    if (p.param3 != 0) {

        const uint NumSubgroups = num_threads / gl_SubgroupSize;
        sum_sq = subgroupAdd(sum_sq);
        if (gl_SubgroupInvocationID == 0) {
            sumsh[gl_SubgroupID] = sum_sq;
        }
        barrier();
        [[unroll]] for (uint s = NumSubgroups / 2; s > 0; s >>= 1) {
            if (gl_SubgroupID < s && gl_SubgroupInvocationID == 0) {
                sum_sq += sumsh[gl_SubgroupID + s];
                sumsh[gl_SubgroupID] = sum_sq;
            }
            barrier();
        }

        if (gl_SubgroupID == 0 && gl_SubgroupInvocationID == 0) {
            partial_sums[orig_idx / (num_iter * num_threads)] = sum_sq;
        }
    }

}
"""


ADD_RMS_F32_F32_F32_NOREPEAT = shader_variant(
    name="add_rms_f32_f32_f32_norepeat",
    family="add_rms_f32_f32_f32_norepeat",
    contract=shader_contract(
        class_name="AddRmsF32F32F32NorepeatProgram",
        shader_name="add_rms_f32_f32_f32_norepeat",
        fields=(
            input_tensor(name="lhs", binding="t_lhs", role="lhs", dtypes=("float32",), shape=("B", "S", "H")),
            input_tensor(name="rhs", binding="t_rhs", role="rhs", dtypes=("float32",), shape=("B", "S", "H")),
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
            src0_name="lhs",
            src1_name="rhs",
            dst_name="output",
            param3=1,
        ),
        dispatch=(1, ceil_div(mul(mul("B", "S"), "H"), 512), 1),
        bindings=(
            storage_buffer_binding(name="t_lhs", binding=0),
            storage_buffer_binding(name="t_rhs", binding=1),
            storage_buffer_binding(name="t_output", binding=2),
            storage_buffer_binding(name="t_partial_sums", binding=3),
        ),
    ),
    specialization_constants={0: 1},
    source=_ADD_SOURCE,
)
