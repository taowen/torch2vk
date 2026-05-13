#version 450
#extension GL_GOOGLE_include_directive : enable
#line 1 "/var/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp"


#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require

#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require


#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_subgroup_extended_types_float16 : require









#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_KHR_shader_subgroup_vote : enable

#line 1 "/var/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/types.glsl"



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























#line 24 "/var/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp"
#line 1 "/var/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl"

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
} pc;




layout(binding = 4) readonly buffer S { float data_s[]; };

layout(binding = 5) writeonly buffer O { float data_o[]; };
layout(binding = 5) writeonly buffer OV4 { vec4 data_ov4[]; };

layout(binding = 6) readonly buffer MO { uint32_t data_mask_opt[]; };





















































































































































































       float16_t perElemOpStoreCol0(const in uint32_t r, const in uint32_t c, const in float16_t elem, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    if (r < N && c == 0) {
        uint32_t offset = iq2 + r;
        data_o[o_offset + offset] = float(elem);
    }
    return elem;
}


       float16_t perElemOpComputeSlope(const in uint32_t r, const in uint32_t c, const in float16_t elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % pc.gqa_ratio);

    uint32_t n_head_log2 = pc.mask_n_head_log2 & 0xFFFF;

    const float16_t base = float16_t(h < n_head_log2 ? pc.m0 : pc.m1);
    const int exph = int(h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1);

    return float16_t(pow(base, float16_t(exph)));
}


       float16_t perElemOpGetSink(const in uint32_t r, const in uint32_t c, const in float16_t elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % pc.gqa_ratio);

    return float16_t(data_s[h]);
}

uint32_t i, N, KV, split_k_index, Tr, start_j, end_j,
         gqa_iq1, iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
         q_stride, k_stride, v_stride, m_stride;

void init_indices()
{
    N = pc.N;
    KV = pc.KV;

    if (pc.k_num > 1) {
        if (pc.gqa_ratio > 1) {
            i = 0;

            gqa_iq1 = gl_WorkGroupID.x / pc.k_num;
            split_k_index = gl_WorkGroupID.x % pc.k_num;
        } else {
            gqa_iq1 = 0;
            split_k_index = gl_WorkGroupID.x % pc.k_num;
            i = gl_WorkGroupID.x / pc.k_num;
        }
    } else if (pc.gqa_ratio > 1) {
        i = 0;
        gqa_iq1 = gl_WorkGroupID.x;
        split_k_index = 0;
    } else {
        i = gl_WorkGroupID.x;
        gqa_iq1 = 0;
        split_k_index = 0;
    }

    Tr = ( ( (N) + (Br) - 1) / (Br));

    start_j = split_k_index * pc.split_kv / Bc;
    end_j = ( ( (min(KV, (split_k_index + 1) * pc.split_kv)) + (Bc) - 1) / (Bc));



    iq2 = gl_WorkGroupID.y * pc.gqa_ratio;
    iq3 = gl_WorkGroupID.z;


    rk2 = pc.neq2 / pc.nek2;
    rk3 = pc.neq3 / pc.nek3;

    rv2 = pc.neq2 / pc.nev2;
    rv3 = pc.neq3 / pc.nev3;


    ik3 = iq3 / rk3;
    ik2 = iq2 / rk2;


    iv3 = iq3 / rv3;
    iv2 = iq2 / rv2;




    q_stride = pc.gqa_ratio > 1 ? (pc.nb02 / 4) : pc.nb01;
    k_stride = pc.nb11;
    v_stride = pc.nb21;




    m_stride = (pc.gqa_ratio > 1) ? (pc.gqa_ratio >> 16) : KV;
}



const float FATTN_KQ_MAX_OFFSET = 3.0f * 0.6931f;



void gqaStore(const in uint32_t r, const in uint32_t c, const in f16vec4 elems, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    uint32_t offset = (iq2 + r) * HSV / 4 + c;
    data_ov4[o_offset + offset] = vec4(elems);
}
#line 25 "/var/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp"

const uint32_t HSK_per_thread = HSK / D_split;
const uint32_t HSV_per_thread = HSV / D_split;

const uint32_t rows_per_thread = Br / row_split;
const uint32_t cols_per_iter = WorkGroupSize / D_split / row_split;
const uint32_t cols_per_thread = Bc / cols_per_iter;
const uint32_t num_subgroups = SubGroupSize == 0 ? 0 : WorkGroupSize / SubGroupSize;


layout(binding = 0) readonly buffer Q { float data_q[]; };
layout(binding = 0) readonly buffer QV4 { vec4 data_qv4[]; };
layout(binding = 1) readonly buffer K { float16_t data_k[]; };
layout(binding = 1) readonly buffer KV4 { f16vec4 data_kv4[]; };
layout(binding = 2) readonly buffer V { float16_t data_v[]; };
layout(binding = 2) readonly buffer VV4 { f16vec4 data_vv4[]; };
layout(binding = 3) readonly buffer M { float16_t data_m[]; };


const uint32_t tmpsh_size = (SubGroupSize > 0) ? (row_split == 1 ? num_subgroups * D_split : num_subgroups) : WorkGroupSize;
shared float tmpsh[tmpsh_size];
shared f16vec4 tmpshv4[tmpsh_size];

const uint32_t masksh_stride = Br + 1;
shared float16_t masksh[Bc * masksh_stride];


const uint32_t qf_stride = HSK / 4 + 1;
shared f16vec4 Qf[Br * qf_stride];







const uint32_t D = HSK > HSV ? HSK : HSV;



const uint32_t kvsh_stride = D / 4 + 1;
shared f16vec4 kvsh[SHMEM_STAGING != 0 ? Bc * kvsh_stride : 1];






shared vec4 occupancy_limiter[LIMIT_OCCUPANCY_SHMEM > 0 ? LIMIT_OCCUPANCY_SHMEM : 1];





void main() {




    init_indices();

    const uint32_t tid = gl_LocalInvocationIndex;
    const uint32_t threads_per_rowgroup = gl_WorkGroupSize.x / row_split;
    const uint32_t row_tid = gl_LocalInvocationIndex / threads_per_rowgroup;
    const uint32_t rowgroup_tid = gl_LocalInvocationIndex % threads_per_rowgroup;
    const uint32_t d_tid = gl_LocalInvocationIndex % D_split;
    const uint32_t col_tid = (gl_LocalInvocationIndex % threads_per_rowgroup) / D_split;

    if (LIMIT_OCCUPANCY_SHMEM > 0) {

        occupancy_limiter[tid] = vec4(tid);

        barrier();

        if (occupancy_limiter[tid] == vec4(99999.0)) {
            data_ov4[0] = vec4(occupancy_limiter[tid]);
        }
    }



    uint32_t q_offset = gqa_iq1 * pc.nb01 + (iq2 * pc.nb02 + iq3 * pc.nb03) / 4;

    [[unroll]] for (uint32_t idx = 0; idx < Br * HSK / 4; idx += gl_WorkGroupSize.x) {
        uint32_t d = (idx + tid) % (HSK / 4);
        uint32_t r = (idx + tid) / (HSK / 4);
        const bool is_in_bounds = r < Br && d < HSK / 4 && i * Br + r < N;

        if (is_in_bounds) {
            Qf[r * qf_stride + d] = f16vec4(data_qv4[q_offset / 4 + (i * Br + r) * q_stride / 4 + d] * pc.scale);
        }




























    }
    barrier();

               f16vec4 Of[rows_per_thread][HSV_per_thread / 4];
    [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++ d) {
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            Of[r][d] = f16vec4(0.0);
        }
    }

    float Lf[rows_per_thread], Mf[rows_per_thread];


    const float NEG_FLT_MAX_OVER_2 = uintBitsToFloat(0xFEFFFFFF);

    [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
        Lf[r] = 0;
        Mf[r] = NEG_FLT_MAX_OVER_2;
    }

           float16_t slope[rows_per_thread];
    [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
        slope[r] = float16_t(1.0);
    }


    if (pc.max_bias > 0.0f) {
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            slope[r] = perElemOpComputeSlope( (row_tid * rows_per_thread + (r)), col_tid, float16_t(0), iq2);
        }
    }

    const uint32_t mo_stride = ( ( (KV) + (16 * Bc) - 1) / (16 * Bc));

    uint32_t mo_offset = mo_stride * i;





    uint32_t k_offset = (ik2 * pc.nb12 + ik3 * pc.nb13) / 2;
    uint32_t v_offset = (iv2 * pc.nb22 + iv3 * pc.nb23) / 2;

    uint32_t m_offset = gqa_iq1 * KV;
    if (pc.nem2 != 1 || pc.nem3 != 1) {
        m_offset += ( (iq3 % pc.nem3) * pc.nem2 + (iq2 % pc.nem2)) * pc.nem1 * KV;
        mo_offset += ( (iq3 % pc.nem3) * pc.nem2 + (iq2 % pc.nem2)) * ( ( (pc.nem1) + (Br) - 1) / (Br)) * mo_stride;
    }

    uint32_t mask_opt = 0;
    uint32_t mask_opt_idx = ~ 0;
    uint32_t mask_opt_bits = 0;

    [[dont_unroll]]
    for (uint32_t j = start_j; j < end_j; ++ j) {
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
                bool nem1_bounds_check = ! (pc.gqa_ratio > 1) && (pc.nem1 % Br) != 0;

                float max_mask = NEG_FLT_MAX_OVER_2;
                barrier();
                [[unroll]] for (uint32_t idx = 0; idx < Bc * Br; idx += gl_WorkGroupSize.x) {
                    uint32_t c = (idx + tid) % Bc;
                    uint32_t r = (idx + tid) / Bc;
                    if (idx + tid < Bc * Br) {
                        if ( (! KV_bounds_check || j * Bc + c < KV) && (! nem1_bounds_check || i * Br + r < pc.nem1)) {
                                     float16_t m = float16_t(data_m[m_offset + (i * Br + r) * m_stride + (j * Bc + c)]);
                            masksh[c * masksh_stride + r] = m;
                            max_mask = max(max_mask, float(m));
                        } else {
                            masksh[c * masksh_stride + r] = float16_t(0);
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

               float16_t Sf[rows_per_thread][cols_per_thread];
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++ c) {
                Sf[r][c] = float16_t(0.0);
            }
        }

        if (SHMEM_STAGING != 0) {
            barrier();

            [[unroll]] for (uint32_t idx = 0; idx < Bc * HSK / 4; idx += gl_WorkGroupSize.x) {
                uint32_t d = (idx + tid) % (HSK / 4);
                uint32_t c = (idx + tid) / (HSK / 4);
                if (idx + gl_WorkGroupSize.x <= Bc * HSK / 4 || c < Bc) {
                               f16vec4 K_Tf = f16vec4(0);
                    if (! KV_bounds_check || j * Bc + c < KV) {






                        K_Tf = f16vec4(data_kv4[k_offset / 4 + (j * Bc + c) * k_stride / 4 + d]);

                    }

                    kvsh[c * kvsh_stride + d] = K_Tf;
                }
            }



















            barrier();
        }




        if (HSK_per_thread / 4 > 4) {
            [[unroll]] for (uint32_t d = 0; d < HSK_per_thread / 4; ++ d) {
                           f16vec4 Q_cache[rows_per_thread];
                [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                    Q_cache[r] = Qf[ (row_tid * rows_per_thread + (r)) * qf_stride + d * D_split + d_tid];
                }

                [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++ c) {
                    if (KV_bounds_check && j * Bc + c * cols_per_iter + col_tid >= KV) {
                        continue;
                    }

                               f16vec4 K_Tf;
                    if (SHMEM_STAGING != 0) {
                        K_Tf = kvsh[ (c * cols_per_iter + col_tid) * kvsh_stride + (d * D_split + d_tid)];
                    } else {






                        K_Tf = f16vec4(data_kv4[k_offset / 4 + (j * Bc + c * cols_per_iter + col_tid) * k_stride / 4 + d * D_split + d_tid]);

                    }
                    [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                        Sf[r][c] += dot(f16vec4(Q_cache[r]), f16vec4(K_Tf));
                    }
                }
            }
        } else {
            [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++ c) {
                if (KV_bounds_check && j * Bc + c * cols_per_iter + col_tid >= KV) {
                    continue;
                }

                [[unroll]] for (uint32_t d = 0; d < HSK_per_thread / 4; ++ d) {
                               f16vec4 K_Tf;
                    if (SHMEM_STAGING != 0) {
                        K_Tf = kvsh[ (c * cols_per_iter + col_tid) * kvsh_stride + (d * D_split + d_tid)];
                    } else {






                        K_Tf = f16vec4(data_kv4[k_offset / 4 + (j * Bc + c * cols_per_iter + col_tid) * k_stride / 4 + d * D_split + d_tid]);

                    }
                    [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                        Sf[r][c] += dot(f16vec4(Qf[ (row_tid * rows_per_thread + (r)) * qf_stride + d * D_split + d_tid]), f16vec4(K_Tf));
                    }
                }
            }
        }









































































































        [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++ c) {

            [[unroll]] for (uint s = D_split / 2; s > 0; s >>= 1) {
                [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                    Sf[r][c] += subgroupShuffleXor(Sf[r][c], s);
                }
            }
        }

        if (LOGIT_SOFTCAP) {
            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++ c) {
                    Sf[r][c] = float16_t(pc.logit_softcap * tanh(Sf[r][c]));
                }
            }
        }

        if (MASK_ENABLE && mask_opt_bits != 2) {
            [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++ c) {
                [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                             float16_t mvf = masksh[ (c * cols_per_iter + col_tid) * masksh_stride + (row_tid * rows_per_thread + (r))];

                    Sf[r][c] += slope[r] * mvf;
                }
            }
        }

        float eMf[rows_per_thread];
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            float rowmaxf = NEG_FLT_MAX_OVER_2;
            [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++ c) {
                if (KV_bounds_check && j * Bc + c * cols_per_iter + col_tid >= KV) {
                    continue;
                }
                rowmaxf = max(rowmaxf, float(Sf[r][c]));
            }
            float Moldf = Mf[r];




            Mf[r] = max(rowmaxf, Moldf);
            eMf[r] = exp(Moldf - Mf[r]);
            Lf[r] = eMf[r] * Lf[r];
        }

        [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++ d) {
            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                Of[r][d] = float16_t(eMf[r]) * Of[r][d];
            }
        }

        if (SHMEM_STAGING != 0) {
            barrier();
            [[unroll]] for (uint32_t idx = 0; idx < Bc * HSV / 4; idx += gl_WorkGroupSize.x) {
                uint32_t d = (idx + tid) % (HSV / 4);
                uint32_t c = (idx + tid) / (HSV / 4);
                if (idx + gl_WorkGroupSize.x <= Bc * HSV / 4 || c < Bc) {
                               f16vec4 V_Tf = f16vec4(0);
                    if (! KV_bounds_check || j * Bc + c < KV) {






                        V_Tf = f16vec4(data_vv4[v_offset / 4 + (j * Bc + c) * v_stride / 4 + d]);

                    }

                    kvsh[c * kvsh_stride + d] = V_Tf;
                }
            }
            barrier();
        }

        [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++ c) {
            if (KV_bounds_check && j * Bc + c * cols_per_iter + col_tid >= KV) {
                continue;
            }

                     float16_t Pf[rows_per_thread];
            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                Pf[r] = float16_t(exp(float(Sf[r][c]) - Mf[r]));
                Lf[r] += Pf[r];
            }

            [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++ d) {
                           f16vec4 Vf;
                if (SHMEM_STAGING != 0) {
                    Vf = kvsh[ (c * cols_per_iter + col_tid) * kvsh_stride + (d * D_split + d_tid)];
                } else {






                    Vf = f16vec4(data_vv4[v_offset / 4 + (j * Bc + c * cols_per_iter + col_tid) * v_stride / 4 + d * D_split + d_tid]);

                }
                [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                    Of[r][d] += f16vec4(Pf[r] * Vf);
                }
            }
        }
    }


    barrier();



    [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
        float rowmaxf = Mf[r];


        if (SubGroupSize > 0) {
            [[unroll]] for (uint s = D_split; s < SubGroupSize; s *= 2) {
                rowmaxf = max(rowmaxf, subgroupShuffleXor(rowmaxf, s));
            }
            if (row_split == 1) {

                barrier();
                if (gl_SubgroupInvocationID == d_tid) {
                    tmpsh[gl_SubgroupID * D_split + d_tid] = rowmaxf;
                }
                barrier();
                rowmaxf = tmpsh[d_tid];
                [[unroll]] for (uint32_t s = 1; s < num_subgroups; ++ s) {
                    rowmaxf = max(rowmaxf, tmpsh[s * D_split + d_tid]);
                }
            }
        } else {
            barrier();
            tmpsh[tid] = rowmaxf;
            barrier();
            [[unroll]] for (int s = int(threads_per_rowgroup) / 2; s >= D_split; s >>= 1) {
                if (rowgroup_tid < s) {
                    tmpsh[tid] = max(tmpsh[tid], tmpsh[tid ^ s]);
                }
                barrier();
            }
            rowmaxf = tmpsh[row_tid * threads_per_rowgroup + d_tid];
        }

        float Moldf = Mf[r];



        Mf[r] = max(rowmaxf, Moldf);
        float eMf = exp(Moldf - Mf[r]);

        Lf[r] = eMf * Lf[r];


        if (SubGroupSize > 0) {
            [[unroll]] for (uint s = D_split; s < SubGroupSize; s *= 2) {
                Lf[r] += subgroupShuffleXor(Lf[r], s);
            }
            if (row_split == 1) {
                barrier();
                if (gl_SubgroupInvocationID == d_tid) {
                    tmpsh[gl_SubgroupID * D_split + d_tid] = Lf[r];
                }
                barrier();
                Lf[r] = tmpsh[d_tid];
                [[unroll]] for (uint32_t s = 1; s < num_subgroups; ++ s) {
                    Lf[r] += tmpsh[s * D_split + d_tid];
                }
            }
        } else {
            barrier();
            tmpsh[tid] = Lf[r];
            barrier();
            [[unroll]] for (int s = int(threads_per_rowgroup) / 2; s >= D_split; s >>= 1) {
                if (rowgroup_tid < s) {
                    tmpsh[tid] = tmpsh[tid] + tmpsh[tid ^ s];
                }
                barrier();
            }
            Lf[r] = tmpsh[row_tid * threads_per_rowgroup + d_tid];
        }

        [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++ d) {
            Of[r][d] = float16_t(eMf) * Of[r][d];

            if (SubGroupSize > 0) {
                [[unroll]] for (uint s = D_split; s < SubGroupSize; s *= 2) {
                    if (! OLD_AMD_WINDOWS) {
                        Of[r][d] += subgroupShuffleXor(Of[r][d], s);
                    } else {



                        Of[r][d] += f16vec4(subgroupShuffleXor(vec4(Of[r][d]), s));
                    }
                }
                if (row_split == 1) {
                    barrier();
                    if (gl_SubgroupInvocationID == d_tid) {
                        tmpshv4[gl_SubgroupID * D_split + d_tid] = Of[r][d];
                    }
                    barrier();
                    Of[r][d] = tmpshv4[d_tid];
                    [[unroll]] for (uint32_t s = 1; s < num_subgroups; ++ s) {
                        Of[r][d] += tmpshv4[s * D_split + d_tid];
                    }
                }
            } else {
                barrier();
                tmpshv4[tid] = Of[r][d];
                barrier();
                [[unroll]] for (int s = int(threads_per_rowgroup) / 2; s >= D_split; s >>= 1) {
                    if (rowgroup_tid < s) {
                        Of[r][d] += tmpshv4[tid ^ s];
                        tmpshv4[tid] = Of[r][d];
                    }
                    barrier();
                }
                Of[r][d] = tmpshv4[row_tid * threads_per_rowgroup + d_tid];
            }
        }
    }




    if (pc.k_num > 1) {
        if (pc.gqa_ratio > 1) {

            uint32_t o_offset = HSV * pc.ne1 * (split_k_index + pc.k_num * (gqa_iq1 + pc.ne2 * iq3)) / 4;

            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                const uint row = (row_tid * rows_per_thread + (r));
                if (row < N) {
                    [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++ d) {
                        gqaStore(row, d * D_split + d_tid, Of[r][d], o_offset, iq2, N);
                    }
                }
            }

            o_offset = HSV * pc.ne1 * pc.k_num * pc.ne2 * pc.ne3 + pc.ne1 * 2 * (split_k_index + pc.k_num * (gqa_iq1 + pc.ne2 * iq3));
            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                const uint row = (row_tid * rows_per_thread + (r));
                if (row < N) {
                    perElemOpStoreCol0(row, 0u, float16_t(Lf[r]), o_offset, iq2, N);
                    perElemOpStoreCol0(row, 0u, float16_t(Mf[r]), o_offset + pc.ne1, iq2, N);
                }
            }
        } else {
            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
                const uint row = (row_tid * rows_per_thread + (r));
                const uint global_row = i * Br + row;

                if (global_row < N) {
                    uint32_t o_offset = HSV * pc.ne1 * (split_k_index + pc.k_num * (global_row + pc.ne2 * iq3)) / 4;

                    [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++ d) {
                        data_ov4[o_offset + iq2 * HSV / 4 + d * D_split + d_tid] = vec4(Of[r][d]);
                    }
                }

                if (global_row < N && d_tid == 0 && col_tid == 0) {
                    uint32_t lm_offset = HSV * pc.ne1 * pc.k_num * pc.ne2 * pc.ne3 + pc.ne1 * 2 * (split_k_index + pc.k_num * (global_row + pc.ne2 * iq3));
                    data_o[lm_offset + iq2] = float(Lf[r]);
                    data_o[lm_offset + pc.ne1 + iq2] = float(Mf[r]);
                }
            }
        }
        return;
    }

    if ( (pc.mask_n_head_log2 & (1 << 24)) != 0) {
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            float sink = perElemOpGetSink( (row_tid * rows_per_thread + (r)), 0u, float16_t(0), iq2);

            float ms = 1.0f;
            float vs = 1.0f;

            if (sink > Mf[r]) {
                ms = exp(Mf[r] - sink);

                [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++ d) {
                    Of[r][d] *= float16_t(ms);
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

    [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++ d) {
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            Of[r][d] *= float16_t(Lfrcp[r]);

            Of[r][d] = clamp(Of[r][d], - float16_t(65504.0), float16_t(65504.0));

        }
    }

    uint32_t o_offset = (gqa_iq1 * pc.ne1 * HSV + iq3 * pc.ne2 * pc.ne1 * HSV) / 4;

    if (pc.gqa_ratio > 1) {
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            const uint row = (row_tid * rows_per_thread + (r));
            if (row < N) {
                [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++ d) {
                    gqaStore(row, d * D_split + d_tid, Of[r][d], o_offset, iq2, N);
                }
            }
        }
    } else {
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++ r) {
            const uint row = (row_tid * rows_per_thread + (r));
            if (i * Br + row < N) {
                [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++ d) {
                    data_ov4[o_offset + (iq2 * HSV + (i * Br + row) * pc.ne1 * HSV) / 4 + d * D_split + d_tid] = vec4(Of[r][d]);
                }
            }
        }
    }
}
