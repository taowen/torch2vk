"""Qwen3 flash attention shader."""

from __future__ import annotations

from torch2vk.shader import (
    Binding,
    BindingAccess,
    PushConstantBlock,
    PushConstantField,
    ShaderContract,
    ShaderVariant,
    TensorContract,
)

_SOURCE = """#version 450
#extension GL_GOOGLE_include_directive : enable
#line 1 "llama_cpp/vulkan-shaders/flash_attn_cm1.comp"

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_cooperative_matrix : enable

#line 1 "llama_cpp/vulkan-shaders/types.glsl"

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

#line 16 "llama_cpp/vulkan-shaders/flash_attn_cm1.comp"
#line 1 "llama_cpp/vulkan-shaders/flash_attn_base.glsl"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint32_t WorkGroupSize = 128;
layout(constant_id = 1) const uint32_t Br = 1;
layout(constant_id = 2) const uint32_t Bc = 32;
layout(constant_id = 3) const uint32_t HSK = 32;
layout(constant_id = 4) const uint32_t HSV = 32;
layout(constant_id = 5) const uint32_t Clamp = 0;
layout(constant_id = 6) const uint32_t D_split = 16;
layout(constant_id = 7) const uint32_t row_split = 1;
layout(constant_id = 8) const uint32_t SubGroupSize = 32;
layout(constant_id = 9) const uint32_t SHMEM_STAGING = 0;
layout(constant_id = 10) const uint32_t Flags = 0;
layout(constant_id = 11) const uint32_t LIMIT_OCCUPANCY_SHMEM = 0;

const bool USE_MASK_OPT = (Flags & 1) != 0;
const bool MASK_ENABLE = (Flags & 2) != 0;
const bool LOGIT_SOFTCAP = (Flags & 4) != 0;
const bool OLD_AMD_WINDOWS = (Flags & 8) != 0;

const uint32_t HSK_pad = (HSK + 15) & ~ 15;
const uint32_t HSV_pad = (HSV + 15) & ~ 15;

const bool KV_bounds_check = Clamp != 0;

layout(push_constant) uniform parameter {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
} p;

layout(binding = 4) readonly buffer S { float data_s[]; };

layout(binding = 5) writeonly buffer O { float data_o[]; };
layout(binding = 5) writeonly buffer OV4 { vec4 data_ov4[]; };

layout(binding = 6) readonly buffer MO { uint32_t data_mask_opt[]; };

       float perElemOpStoreCol0(const in uint32_t r, const in uint32_t c, const in float elem, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    if (r < N && c == 0) {
        uint32_t offset = iq2 + r;
        data_o[o_offset + offset] = float(elem);
    }
    return elem;
}

       float perElemOpComputeSlope(const in uint32_t r, const in uint32_t c, const in float elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    uint32_t n_head_log2 = p.mask_n_head_log2 & 0xFFFF;

    const float base = float(h < n_head_log2 ? p.m0 : p.m1);
    const int exph = int(h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1);

    return float(pow(base, float(exph)));
}

       float perElemOpGetSink(const in uint32_t r, const in uint32_t c, const in float elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    return float(data_s[h]);
}

uint32_t i, N, KV, split_k_index, Tr, start_j, end_j,
         gqa_iq1, iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
         q_stride, k_stride, v_stride, m_stride;

void init_indices()
{
    N = p.N;
    KV = p.KV;

    if (p.k_num > 1) {
        if (p.gqa_ratio > 1) {
            i = 0;

            gqa_iq1 = gl_WorkGroupID.x / p.k_num;
            split_k_index = gl_WorkGroupID.x % p.k_num;
        } else {
            gqa_iq1 = 0;
            split_k_index = gl_WorkGroupID.x % p.k_num;
            i = gl_WorkGroupID.x / p.k_num;
        }
    } else if (p.gqa_ratio > 1) {
        i = 0;
        gqa_iq1 = gl_WorkGroupID.x;
        split_k_index = 0;
    } else {
        i = gl_WorkGroupID.x;
        gqa_iq1 = 0;
        split_k_index = 0;
    }

    Tr = ( ( (N) + (Br) - 1) / (Br));

    start_j = split_k_index * p.split_kv / Bc;
    end_j = ( ( (min(KV, (split_k_index + 1) * p.split_kv)) + (Bc) - 1) / (Bc));

    iq2 = gl_WorkGroupID.y * p.gqa_ratio;
    iq3 = gl_WorkGroupID.z;

    rk2 = p.neq2 / p.nek2;
    rk3 = p.neq3 / p.nek3;

    rv2 = p.neq2 / p.nev2;
    rv3 = p.neq3 / p.nev3;

    ik3 = iq3 / rk3;
    ik2 = iq2 / rk2;

    iv3 = iq3 / rv3;
    iv2 = iq2 / rv2;

    q_stride = p.gqa_ratio > 1 ? (p.nb02 / 4) : p.nb01;
    k_stride = p.nb11;
    v_stride = p.nb21;

    m_stride = (p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : KV;
}

const float FATTN_KQ_MAX_OFFSET = 3.0f * 0.6931f;

void gqaStore(const in uint32_t r, const in uint32_t c, const in f16vec4 elems, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    uint32_t offset = (iq2 + r) * HSV / 4 + c;
    data_ov4[o_offset + offset] = vec4(elems);
}
#line 17 "llama_cpp/vulkan-shaders/flash_attn_cm1.comp"

const uint32_t MatBr = 16;
const uint32_t MatBc = 16;

const uint32_t rows_per_thread = Br / row_split;
const uint32_t cols_per_iter = gl_WorkGroupSize.x / row_split;
const uint32_t cols_per_thread = Bc / cols_per_iter;

layout(binding = 0) readonly buffer Q { float data_q[]; };
layout(binding = 0) readonly buffer QV4 { vec4 data_qv4[]; };
layout(binding = 1) readonly buffer K { float16_t data_k[]; };
layout(binding = 1) readonly buffer KV4 { f16vec4 data_kv4[]; };
layout(binding = 2) readonly buffer V { float16_t data_v[]; };
layout(binding = 2) readonly buffer VV4 { f16vec4 data_vv4[]; };
layout(binding = 3) readonly buffer M { float16_t data_m[]; };

shared float tmpsh[row_split];

const uint32_t qstride = HSK_pad / 4 + 2;
shared f16vec4 Qf[Br * qstride];

const uint psh_stride = Br / 4 + 2;
shared f16vec4 Psh[Bc * psh_stride];

const uint32_t sfshstride = (HSK <= 128) ? (Br / 4 + 2) : Br / 4;
shared vec4 sfsh[Bc * sfshstride];

const uint32_t D_pad = HSK_pad > HSV_pad ? HSK_pad : HSV_pad;
const uint32_t kvsh_stride = (SHMEM_STAGING != 0 ? D_pad : MatBr) / 4 + 2;
const uint v_cols = MatBc / 4 * row_split;
const uint vsh_stride = v_cols;
shared f16vec4 kvsh[ (kvsh_stride >= vsh_stride) ? (Bc * kvsh_stride) : (Bc * vsh_stride)];

const uint32_t osh_stride = row_split * MatBr / 4;
shared f16vec4 pvsh[MatBc * osh_stride];

shared float slope[Br];

void main() {

    init_indices();

    const uint32_t tid = gl_LocalInvocationIndex;

    const uint32_t threads_per_rowgroup = gl_WorkGroupSize.x / row_split;
    const uint32_t d_per_thread = (HSV / 4 + threads_per_rowgroup - 1) / threads_per_rowgroup;
    const uint32_t row_tid = gl_LocalInvocationIndex / threads_per_rowgroup;
    const uint32_t col_tid = gl_LocalInvocationIndex % threads_per_rowgroup;

    if ( (HSK % 16) != 0) {
        [[unroll]] for (uint i = 0; i < Br * qstride; i += gl_WorkGroupSize.x) {
            if (i + tid < Br * qstride) {
                Qf[i + tid] = f16vec4(0);
            }
        }
        barrier();
    }

    uint32_t q_offset = gqa_iq1 * p.nb01 + (iq2 * p.nb02 + iq3 * p.nb03) / 4;

    [[unroll]] for (uint32_t idx = 0; idx < Br * HSK / 4; idx += gl_WorkGroupSize.x) {
        uint32_t d = (idx + tid) % (HSK / 4);
        uint32_t r = (idx + tid) / (HSK / 4);
        if (r < Br && d < HSK / 4 &&
            i * Br + r < N) {
            Qf[r * qstride + d] = f16vec4(data_qv4[q_offset / 4 + (i * Br + r) * q_stride / 4 + d] * p.scale);
        }
    }
    barrier();

    f16vec4 Of[rows_per_thread][d_per_thread];
    [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
        [[unroll]] for (uint32_t d = 0; d < d_per_thread; ++ d) {
            Of[r][d] = f16vec4(0.0);
        }
    }

    float Lf[rows_per_thread], Mf[rows_per_thread];

    const float NEG_FLT_MAX_OVER_2 = uintBitsToFloat(0xFEFFFFFF);

    [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
        Lf[r] = 0;
        Mf[r] = NEG_FLT_MAX_OVER_2;
    }

    if (p.max_bias > 0.0f) {
        if (tid < Br) {
            uint r = tid;
            slope[r] = perElemOpComputeSlope(r, col_tid, float(0), iq2);
        }
    } else {
        if (tid < Br) {
            uint r = tid;
            slope[r] = float(1.0);
        }
    }

    const uint32_t mo_stride = ( ( (KV) + (16 * Bc) - 1) / (16 * Bc));

    uint32_t mo_offset = mo_stride * i;

    uint32_t k_offset = (ik2 * p.nb12 + ik3 * p.nb13) / 2;
    uint32_t v_offset = (iv2 * p.nb22 + iv3 * p.nb23) / 2;

    uint32_t m_offset = gqa_iq1 * KV;
    if (p.nem2 != 1 || p.nem3 != 1) {
        m_offset += ( (iq3 % p.nem3) * p.nem2 + (iq2 % p.nem2)) * p.nem1 * KV;
        mo_offset += ( (iq3 % p.nem3) * p.nem2 + (iq2 % p.nem2)) * ( ( (p.nem1) + (Br) - 1) / (Br)) * mo_stride;
    }

    uint32_t mask_opt = 0;
    uint32_t mask_opt_idx = ~ 0;
    uint32_t mask_opt_bits = 0;
    f16vec4 mask_cache[Bc * Br / 4 / WorkGroupSize];

    [[dont_unroll]]
    for (uint32_t j = start_j; j < end_j; ++ j) {

        [[unroll]] for (uint32_t idx = 0; idx < mask_cache.length(); ++ idx) {
            mask_cache[idx] = f16vec4(0);
        }

        if (MASK_ENABLE) {
            if (USE_MASK_OPT && mask_opt_idx != j / 16) {
                mask_opt_idx = j / 16;
                mask_opt = data_mask_opt[mo_offset + mask_opt_idx];
            }
            mask_opt_bits = (mask_opt >> ( (j % 16) * 2)) & 0x3;
            if (mask_opt_bits == 1) {

                continue;
            }

            if (mask_opt_bits != 2) {
                bool nem1_bounds_check = ! (p.gqa_ratio > 1) && (p.nem1 % Br) != 0;

                float max_mask = NEG_FLT_MAX_OVER_2;
                [[unroll]] for (uint32_t idx = 0; idx < Bc * Br / 4; idx += gl_WorkGroupSize.x) {
                    uint32_t c = (idx + tid) / (Br / 4);
                    uint32_t r = (idx + tid) % (Br / 4);
                    if (idx + tid < Bc * Br / 4 || idx + gl_WorkGroupSize.x <= Bc * Br / 4) {
                        if ( (! KV_bounds_check || j * Bc + c < KV)) {
                            f16vec4 m;
                            if (! nem1_bounds_check || i * Br + r * 4 + 3 < p.nem1) {
                                m = f16vec4(data_m[m_offset + (i * Br + r * 4) * m_stride + (j * Bc + c)],
                                            data_m[m_offset + (i * Br + r * 4 + 1) * m_stride + (j * Bc + c)],
                                            data_m[m_offset + (i * Br + r * 4 + 2) * m_stride + (j * Bc + c)],
                                            data_m[m_offset + (i * Br + r * 4 + 3) * m_stride + (j * Bc + c)]);
                                max_mask = max(max(max(max(max_mask, float(m[0])), float(m[1])), float(m[2])), float(m[3]));
                            } else if (i * Br + r * 4 + 2 < p.nem1) {
                                m = f16vec4(data_m[m_offset + (i * Br + r * 4) * m_stride + (j * Bc + c)],
                                            data_m[m_offset + (i * Br + r * 4 + 1) * m_stride + (j * Bc + c)],
                                            data_m[m_offset + (i * Br + r * 4 + 2) * m_stride + (j * Bc + c)],
                                            0.0);
                                max_mask = max(max(max(max_mask, float(m[0])), float(m[1])), float(m[2]));
                            } else if (i * Br + r * 4 + 1 < p.nem1) {
                                m = f16vec4(data_m[m_offset + (i * Br + r * 4) * m_stride + (j * Bc + c)],
                                            data_m[m_offset + (i * Br + r * 4 + 1) * m_stride + (j * Bc + c)],
                                            0.0,
                                            0.0);
                                max_mask = max(max(max_mask, float(m[0])), float(m[1]));
                            } else if (i * Br + r * 4 < p.nem1) {
                                m = f16vec4(data_m[m_offset + (i * Br + r * 4) * m_stride + (j * Bc + c)],
                                            0.0,
                                            0.0,
                                            0.0);
                                max_mask = max(max_mask, float(m[0]));
                            } else {
                                m = f16vec4(0.0);
                            }
                            mask_cache[idx / WorkGroupSize] = m;
                        }
                    }
                }

                bool all_less = subgroupAll(max_mask <= NEG_FLT_MAX_OVER_2);
                barrier();
                if (gl_SubgroupInvocationID == 0) {
                    tmpsh[gl_SubgroupID] = all_less ? NEG_FLT_MAX_OVER_2 : 0.0f;
                }
                barrier();
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++ s) {
                    max_mask = max(max_mask, tmpsh[s]);
                }
                if (max_mask <= NEG_FLT_MAX_OVER_2) {
                    continue;
                }
            }
        }

        if (SHMEM_STAGING != 0) {
            [[unroll]] for (uint32_t idx = 0; idx < Bc * HSK_pad / 4; idx += gl_WorkGroupSize.x) {
                uint32_t d = (idx + tid) % (HSK_pad / 4);
                uint32_t c = (idx + tid) / (HSK_pad / 4);
                if (idx + gl_WorkGroupSize.x <= Bc * HSK_pad / 4 || c < Bc) {
                    f16vec4 K_Tf = f16vec4(0);
                    if ( (! KV_bounds_check || j * Bc + c < KV) && (HSK == HSK_pad || d < HSK / 4)) {

                        K_Tf = f16vec4(data_kv4[k_offset / 4 + (j * Bc + c) * k_stride / 4 + d]);

                    }

                    kvsh[c * kvsh_stride + d] = K_Tf;
                }
            }
            barrier();
        }

        coopmat < float, gl_ScopeSubgroup, MatBc, MatBr, gl_MatrixUseAccumulator > SfMat = coopmat < float, gl_ScopeSubgroup, MatBc, MatBr, gl_MatrixUseAccumulator > (0);
        coopmat < float16_t, gl_ScopeSubgroup, MatBc, 16, gl_MatrixUseA > KMat;
        coopmat < float16_t, gl_ScopeSubgroup, 16, MatBr, gl_MatrixUseB > QMat;

        [[unroll]] for (uint32_t d = 0; d < HSK_pad / 16; ++ d) {

            if (SHMEM_STAGING == 0) {

            if (KV_bounds_check || d * 16 + 16 > HSK) {

            barrier();
            [[unroll]] for (uint32_t idx = 0; idx < Bc * MatBr / 4; idx += gl_WorkGroupSize.x) {
                uint32_t col_vec = (idx + tid) % (MatBr / 4);
                uint32_t row = (idx + tid) / (MatBr / 4);
                if (idx + tid < Bc * MatBr / 4) {
                    f16vec4 K_Tf = f16vec4(0);
                    if ( (! KV_bounds_check || j * Bc + row < KV) && (HSK == HSK_pad || d * 16 + col_vec * 4 < HSK)) {

                        K_Tf = f16vec4(data_kv4[k_offset / 4 + (j * Bc + row) * k_stride / 4 + d * 16 / 4 + col_vec]);

                    }

                    kvsh[row * kvsh_stride + col_vec] = K_Tf;
                }
            }
            barrier();

            }

            if (KV_bounds_check || d * 16 + 16 > HSK)

            {
                uint coord = (gl_SubgroupID * MatBc) * kvsh_stride;
                coopMatLoad(KMat, kvsh, coord, kvsh_stride, gl_CooperativeMatrixLayoutRowMajor);
            }

            else {
                const uint coord = k_offset / 4 + (j * Bc + gl_SubgroupID * MatBc) * k_stride / 4 + d * 16 / 4;
                coopMatLoad(KMat, data_kv4, coord, k_stride / 4, gl_CooperativeMatrixLayoutRowMajor);
            }

            } else {
                uint coord = (gl_SubgroupID * MatBc) * kvsh_stride + d * 16 / 4;
                coopMatLoad(KMat, kvsh, coord, kvsh_stride, gl_CooperativeMatrixLayoutRowMajor);
            }

            coopMatLoad(QMat, Qf, d * 16 / 4, qstride, gl_CooperativeMatrixLayoutColumnMajor);

            SfMat = coopMatMulAdd(KMat, QMat, SfMat);
        }

        uint coord = gl_SubgroupID * MatBc * sfshstride;
        coopMatStore(SfMat, sfsh, coord, sfshstride, gl_CooperativeMatrixLayoutRowMajor);
        barrier();

        if (LOGIT_SOFTCAP) {
            [[unroll]] for (uint32_t idx = 0; idx < Bc * Br / 4; idx += gl_WorkGroupSize.x) {
                uint32_t c = (idx + tid) / (Br / 4);
                uint32_t r = (idx + tid) % (Br / 4);
                if (idx + tid < Bc * Br / 4 || idx + gl_WorkGroupSize.x <= Bc * Br / 4) {
                    sfsh[c * sfshstride + r] = vec4(p.logit_softcap * tanh(sfsh[c * sfshstride + r]));
                }
            }
            barrier();
        }

        if (MASK_ENABLE && mask_opt_bits != 2) {
            [[unroll]] for (uint32_t idx = 0; idx < Bc * Br / 4; idx += gl_WorkGroupSize.x) {
                uint32_t c = (idx + tid) / (Br / 4);
                uint32_t r = (idx + tid) % (Br / 4);
                if (idx + tid < Bc * Br / 4 || idx + gl_WorkGroupSize.x <= Bc * Br / 4) {
                    if (! KV_bounds_check || j * Bc + c < KV) {

                                 vec4 masks = vec4(mask_cache[idx / WorkGroupSize]);
                                 vec4 slopes = vec4(slope[r * 4], slope[r * 4 + 1], slope[r * 4 + 2], slope[r * 4 + 3]);
                        sfsh[c * sfshstride + r] += slopes * masks;
                    }
                }
            }
            barrier();
        }

        float eMf[rows_per_thread];
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            const uint r_vec = (row_tid * rows_per_thread + (r)) / 4;
            const uint r_comp = (row_tid * rows_per_thread + (r)) % 4;

            float rowmaxf = NEG_FLT_MAX_OVER_2;
            [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++ c) {
                if (KV_bounds_check && j * Bc + c * cols_per_iter + col_tid >= KV) {
                    continue;
                }
                rowmaxf = max(rowmaxf, float(sfsh[r_vec + (c * cols_per_iter + col_tid) * sfshstride][r_comp]));
            }
            float Moldf = Mf[r];

            rowmaxf = subgroupMax(rowmaxf);

            Mf[r] = max(rowmaxf, Moldf);
            eMf[r] = exp(Moldf - Mf[r]);

            Lf[r] = eMf[r] * Lf[r];
        }

        [[unroll]] for (uint32_t d0 = 0; d0 < HSV / 4; d0 += threads_per_rowgroup) {
            const uint d_local = d0 / threads_per_rowgroup;
            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                Of[r][d_local] = float16_t(eMf[r]) * Of[r][d_local];
            }
        }

        [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++ c) {
            const uint col = c * cols_per_iter + col_tid;

            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; r += 4) {
                const uint row = (row_tid * rows_per_thread + (r));
                if (KV_bounds_check && j * Bc + col >= KV) {
                    Psh[col * psh_stride + row / 4] = f16vec4(0.0f);
                } else {
                    const vec4 mfvec = vec4(Mf[r], Mf[r + 1], Mf[r + 2], Mf[r + 3]);
                    const f16vec4 Pf = f16vec4(exp(vec4(sfsh[row / 4 + col * sfshstride]) - mfvec));
                    [[unroll]] for (uint32_t vec_idx = 0; vec_idx < 4; ++ vec_idx) {
                        Lf[r + vec_idx] += Pf[vec_idx];
                    }
                    Psh[col * psh_stride + row / 4] = Pf;
                }
            }
        }

        if (SHMEM_STAGING != 0) {
            [[unroll]] for (uint32_t idx = 0; idx < Bc * HSV_pad / 4; idx += gl_WorkGroupSize.x) {
                uint32_t d = (idx + tid) % (HSV_pad / 4);
                uint32_t c = (idx + tid) / (HSV_pad / 4);
                if (idx + gl_WorkGroupSize.x <= Bc * HSV_pad / 4 || c < Bc) {
                    f16vec4 V_Tf = f16vec4(0);
                    if ( (! KV_bounds_check || j * Bc + c < KV) && (HSV == HSV_pad || d < HSV / 4)) {

                        V_Tf = f16vec4(data_vv4[v_offset / 4 + (j * Bc + c) * v_stride / 4 + d]);

                    }

                    kvsh[c * kvsh_stride + d] = V_Tf;
                }
            }
        }
        barrier();

        const uint num_hsv_tiles = (HSV + MatBc * row_split - 1) / (MatBc * row_split);

        [[unroll]] for (uint32_t hsv_tile = 0; hsv_tile < num_hsv_tiles; ++ hsv_tile) {
            const uint hsv_offset = (hsv_tile * row_split + gl_SubgroupID) * 16;

            coopmat < float16_t, gl_ScopeSubgroup, MatBc, MatBr, gl_MatrixUseAccumulator > PVMat = coopmat < float16_t, gl_ScopeSubgroup, MatBc, MatBr, gl_MatrixUseAccumulator > (0);

            const uint v_rows = Bc;
            const uint v_total = v_rows * v_cols;
            const uint v_loads_per_thread = v_total / gl_WorkGroupSize.x;

            if (SHMEM_STAGING == 0) {

            if (KV_bounds_check) {

            [[unroll]] for (uint32_t i = 0; i < v_loads_per_thread; ++ i) {
                const uint idx = i * gl_WorkGroupSize.x + tid;
                const uint row = idx / v_cols;
                const uint col = idx % v_cols;

                const uint v_row = j * Bc + row;
                const uint v_col = hsv_tile * MatBc * row_split + col * 4;

                const uint coord = v_row * v_stride * 1 + v_col;
                const uint ib = coord / 1;
                const uint iqs = coord % 1;

                if (! KV_bounds_check || (v_row < KV && v_col < HSV)) {

                    kvsh[row * vsh_stride + col] = data_vv4[ (v_offset + v_row * v_stride + v_col) / 4];

                } else {
                    kvsh[row * vsh_stride + col] = f16vec4(0.0f);
                }
            }

            }

            }
            barrier();

            const uint o_offset = gl_SubgroupID * MatBr / 4;

            if (hsv_offset < HSV_pad) {
                [[unroll]] for (uint32_t bc_chunk = 0; bc_chunk < Bc / MatBc; ++ bc_chunk) {
                    coopMatLoad(KMat, Psh, bc_chunk * MatBc * psh_stride, psh_stride, gl_CooperativeMatrixLayoutColumnMajor);

                    if (SHMEM_STAGING == 0) {

                    if (! KV_bounds_check) {

                        const uint v_tile_row = j * Bc + bc_chunk * MatBc;
                        const uint v_tile_offset = v_offset / 4 + v_tile_row * v_stride / 4 + hsv_offset / 4;
                        coopMatLoad(QMat, data_vv4, v_tile_offset, v_stride / 4, gl_CooperativeMatrixLayoutRowMajor);
                    } else

                    {
                        const uint v_tile_offset = bc_chunk * MatBr * v_cols + gl_SubgroupID * (MatBc / 4);
                        coopMatLoad(QMat, kvsh, v_tile_offset, vsh_stride, gl_CooperativeMatrixLayoutRowMajor);
                    }
                    } else {
                        const uint v_tile_offset = bc_chunk * MatBc * kvsh_stride + (hsv_tile * row_split + gl_SubgroupID) * (MatBc / 4);
                        coopMatLoad(QMat, kvsh, v_tile_offset, kvsh_stride, gl_CooperativeMatrixLayoutRowMajor);
                    }

                    PVMat = coopMatMulAdd(KMat, QMat, PVMat);
                }

                coopMatStore(PVMat, pvsh, o_offset, osh_stride, gl_CooperativeMatrixLayoutRowMajor);
            }

            barrier();

            const uint hsv_per_tile = row_split * MatBc;
            const uint hsv_base = hsv_tile * hsv_per_tile;
            const uint d_values_per_tile = hsv_per_tile / 4;

            const uint d_start = hsv_tile * d_values_per_tile;
            const uint d_end = min(d_start + d_values_per_tile, HSV / 4);

            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                const uint row = (row_tid * rows_per_thread + (r));

                [[unroll]] for (uint32_t d_local = 0; d_local < d_per_thread; ++ d_local) {
                    const uint d = d_local * threads_per_rowgroup + col_tid;
                    const uint hsv_col = 4 * d;

                    if (hsv_col >= hsv_base && hsv_col < hsv_base + hsv_per_tile && hsv_col < HSV) {
                        const uint local_hsv = (hsv_col - hsv_base) / 4;
                        Of[r][d_local] += pvsh[row * osh_stride + local_hsv];
                    }
                }
            }
        }

        barrier();
    }

    [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
        Lf[r] = subgroupAdd(Lf[r]);
    }

    if (p.k_num > 1) {
        if (p.gqa_ratio > 1) {

            uint32_t o_offset = HSV * p.ne1 * (split_k_index + p.k_num * (gqa_iq1 + p.ne2 * iq3)) / 4;

            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                if ( (row_tid * rows_per_thread + (r)) < N) {
                    [[unroll]] for (uint32_t d0 = 0; d0 < HSV / 4; d0 += threads_per_rowgroup) {
                        const uint d = d0 + col_tid;
                        if (d >= HSV / 4) break;
                        const uint d_local = d0 / threads_per_rowgroup;
                        gqaStore( (row_tid * rows_per_thread + (r)), d, Of[r][d_local], o_offset, iq2, N);
                    }
                }
            }

            o_offset = HSV * p.ne1 * p.k_num * p.ne2 * p.ne3 + p.ne1 * 2 * (split_k_index + p.k_num * (gqa_iq1 + p.ne2 * iq3));
            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                if ( (row_tid * rows_per_thread + (r)) < N) {
                    perElemOpStoreCol0( (row_tid * rows_per_thread + (r)), 0u, float(Lf[r]), o_offset, iq2, N);
                    perElemOpStoreCol0( (row_tid * rows_per_thread + (r)), 0u, float(Mf[r]), o_offset + p.ne1, iq2, N);
                }
            }
        } else {
            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                const uint row = (row_tid * rows_per_thread + (r));
                const uint global_row = i * Br + row;

                if (global_row < N) {
                    uint32_t o_offset = HSV * p.ne1 * (split_k_index + p.k_num * (global_row + p.ne2 * iq3)) / 4;

                    [[unroll]] for (uint32_t d0 = 0; d0 < HSV / 4; d0 += threads_per_rowgroup) {
                        const uint d = d0 + col_tid;
                        if (d >= HSV / 4) break;
                        data_ov4[o_offset + iq2 * HSV / 4 + d] = vec4(Of[r][d / threads_per_rowgroup]);
                    }
                }

                if (global_row < N && col_tid == 0) {
                    uint32_t lm_offset = HSV * p.ne1 * p.k_num * p.ne2 * p.ne3 + p.ne1 * 2 * (split_k_index + p.k_num * (global_row + p.ne2 * iq3));
                    data_o[lm_offset + iq2] = float(Lf[r]);
                    data_o[lm_offset + p.ne1 + iq2] = float(Mf[r]);
                }
            }
        }

        return;
    }

    if ( (p.mask_n_head_log2 & (1 << 24)) != 0) {
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            float sink = perElemOpGetSink( (row_tid * rows_per_thread + (r)), 0u, float(0), iq2);

            float ms = 1.0f;
            float vs = 1.0f;

            if (sink > Mf[r]) {
                ms = exp(Mf[r] - sink);

                [[unroll]] for (uint32_t d0 = 0; d0 < HSV / 4; d0 += threads_per_rowgroup) {
                    const uint d_local = d0 / threads_per_rowgroup;
                    Of[r][d_local] *= float16_t(ms);
                }
            } else {
                vs = exp(sink - Mf[r]);
            }

            Lf[r] = Lf[r] * ms + vs;
        }
    }

    float Lfrcp[rows_per_thread];
    [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
        Lfrcp[r] = (Lf[r] == 0.0) ? 0.0 : (1.0 / Lf[r]);
    }

    [[unroll]] for (uint32_t d0 = 0; d0 < HSV / 4; d0 += threads_per_rowgroup) {
        const uint d_local = d0 / threads_per_rowgroup;
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            Of[r][d_local] *= float16_t(Lfrcp[r]);

            Of[r][d_local] = clamp(Of[r][d_local], - float16_t(65504.0), float16_t(65504.0));

        }
    }

    uint32_t o_offset = (gqa_iq1 * p.ne1 * HSV + iq3 * p.ne2 * p.ne1 * HSV) / 4;

    if (p.gqa_ratio > 1) {
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            if ( (row_tid * rows_per_thread + (r)) < N) {
                [[unroll]] for (uint32_t d0 = 0; d0 < HSV / 4; d0 += threads_per_rowgroup) {
                    const uint d = d0 + col_tid;
                    if (d >= HSV / 4) break;
                    const uint d_local = d0 / threads_per_rowgroup;
                    gqaStore( (row_tid * rows_per_thread + (r)), d, Of[r][d_local], o_offset, iq2, N);
                }
            }
        }
    } else {
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            if (i * Br + (row_tid * rows_per_thread + (r)) < N) {
                [[unroll]] for (uint32_t d0 = 0; d0 < HSV / 4; d0 += threads_per_rowgroup) {
                    const uint d = d0 + col_tid;
                    if (d >= HSV / 4) break;
                    const uint d_local = d0 / threads_per_rowgroup;
                    data_ov4[o_offset + (iq2 * HSV + (i * Br + (row_tid * rows_per_thread + (r))) * p.ne1 * HSV) / 4 + d] = vec4(Of[r][d_local]);
                }
            }
        }
    }
}
"""

FLASH_ATTN_F32_F16 = ShaderVariant(
    name="flash_attn_f32_f16_aligned_f32accf16",
    family="flash_attention",
    contract=ShaderContract(
        name="flash_attn_f32_f16_aligned_f32accf16",
        inputs={
            "q": TensorContract(dtype="float32", shape=("B", "S", "QH", "D")),
            "k": TensorContract(dtype="float16", shape=("B", "T", "KH", "D")),
            "v": TensorContract(dtype="float16", shape=("B", "T", "KH", "D")),
            "mask": TensorContract(dtype="float16", shape=("B", 1, "S", "T")),
            "sinks_placeholder": TensorContract(dtype="float32", shape=("B", "S", "QH", "D")),
            "mask_opt_placeholder": TensorContract(dtype="float32", shape=("B", "S", "QH", "D")),
        },
        outputs={"split_k_output": TensorContract(dtype="float32", shape=("SK",))},
        bindings=(
            Binding("q", 0, BindingAccess.READ),
            Binding("k", 1, BindingAccess.READ),
            Binding("v", 2, BindingAccess.READ),
            Binding("mask", 3, BindingAccess.READ),
            Binding("sinks_placeholder", 4, BindingAccess.READ),
            Binding("split_k_output", 5, BindingAccess.WRITE),
            Binding("mask_opt_placeholder", 6, BindingAccess.READ),
        ),
        dispatch=("S*4", "QH", "B"),
        push_constants=PushConstantBlock(
            size=128,
            fields=(
                PushConstantField("N", 0, "uint32", "S"),
                PushConstantField("KV", 4, "uint32", "T"),
                PushConstantField("ne1", 8, "uint32", "QH"),
                PushConstantField("ne2", 12, "uint32", "S"),
                PushConstantField("ne3", 16, "uint32", "B"),
                PushConstantField("neq2", 20, "uint32", "QH"),
                PushConstantField("neq3", 24, "uint32", "B"),
                PushConstantField("nek2", 28, "uint32", "KH"),
                PushConstantField("nek3", 32, "uint32", "B"),
                PushConstantField("nev2", 36, "uint32", "KH"),
                PushConstantField("nev3", 40, "uint32", "B"),
                PushConstantField("nem1", 44, "uint32", "S"),
                PushConstantField("nem2", 48, "uint32", 1),
                PushConstantField("nem3", 52, "uint32", "B"),
                PushConstantField("nb01", 56, "uint32", "D*QH"),
                PushConstantField("nb02", 60, "uint32", "D*4"),
                PushConstantField("nb03", 64, "uint32", "D*QH*S*4"),
                PushConstantField("nb11", 68, "uint32", "D*KH"),
                PushConstantField("nb12", 72, "uint32", "D*2"),
                PushConstantField("nb13", 76, "uint32", "D*KH*T*2"),
                PushConstantField("nb21", 80, "uint32", "D*KH"),
                PushConstantField("nb22", 84, "uint32", "D*2"),
                PushConstantField("nb23", 88, "uint32", "D*KH*T*2"),
                PushConstantField("scale", 92, "float32", 0.08838834764831845),
                PushConstantField("max_bias", 96, "float32", 0.0),
                PushConstantField("logit_softcap", 100, "float32", 0.0),
                PushConstantField("mask_n_head_log2", 104, "uint32", 1),
                PushConstantField("m0", 108, "float32", 0.0),
                PushConstantField("m1", 112, "float32", 0.0),
                PushConstantField("gqa_ratio", 116, "uint32", 1),
                PushConstantField("split_kv", 120, "uint32", 4),
                PushConstantField("k_num", 124, "uint32", 4),
            ),
        ),
    ),
    specialization_constants={
        0: 256,
        1: 16,
        2: 64,
        3: 128,
        4: 128,
        5: 1,
        6: 8,
        7: 4,
        8: 64,
        9: 0,
        10: 2,
        11: 0,
    },
    compile_defines=(
        "ACC_TYPE=float",
        "ACC_TYPEV2=vec2",
        "ACC_TYPEV4=vec4",
        "COOPMAT=1",
        "D_TYPE=float",
        "D_TYPEV4=vec4",
        "FLOAT16=1",
        "FLOAT_TYPE=float16_t",
        "FLOAT_TYPEV2=f16vec2",
        "FLOAT_TYPEV4=f16vec4",
        "FLOAT_TYPE_MAX=float16_t(65504.0)",
        "Q_TYPE=float",
    ),
    source=_SOURCE,
)
