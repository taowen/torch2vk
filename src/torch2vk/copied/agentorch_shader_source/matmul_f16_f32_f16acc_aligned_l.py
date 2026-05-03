"""llama.cpp runtime-matched F16 x F32 cooperative prefill matmul."""

from __future__ import annotations

from collections.abc import Mapping

from agentorch.types import DeviceTensorLike as DeviceTensor
from agentorch.kernel.contract import (
    PushConstantSpec,
    ceil_div,
    input_tensor,
    mul,
    output_tensor,
    push_constant_block,
    push_uint32,
    shader_contract,
    storage_buffer_binding,
)

from .shader_execution_requirements import CooperativeMatrixRequirements, ShaderExecutionRequirements
from .shader_variant import shader_variant


_MATMUL_F16_F32_F16ACC_ALIGNED_CM1_SOURCE = """\
#version 450
#extension GL_GOOGLE_include_directive : enable
#line 1 "/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp"


#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require


#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require










#extension GL_KHR_cooperative_matrix : enable
#extension GL_KHR_memory_scope_semantics : enable



#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable






#line 1 "/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/types.glsl"



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























#line 32 "/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp"
























layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer A { f16mat2x4 data_a[]; };







layout(binding = 1) readonly buffer B { mat2x4 data_b[]; };
layout(binding = 2) writeonly buffer D { float data_d[]; };






layout(push_constant) uniform parameter
{
    uint M;
    uint N;
    uint K;
    uint stride_a;
    uint stride_b;
    uint stride_d;

    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;







    uint base_work_group_z;
    uint num_batches;
    uint k_split;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;

} p;

layout(constant_id = 0) const uint BLOCK_SIZE = 64;
layout(constant_id = 1) const uint BM = 64;
layout(constant_id = 2) const uint BN = 64;
layout(constant_id = 4) const uint WM = 32;
layout(constant_id = 5) const uint WN = 32;
layout(constant_id = 6) const uint WMITER = 2;
layout(constant_id = 7) const uint TM = 4;
layout(constant_id = 8) const uint TN = 2;
layout(constant_id = 9) const uint TK = 1;
layout(constant_id = 10) const uint WARP = 32;















shared f16vec2 buf_a[BM * (32 / 2 + 4)];
shared f16vec2 buf_b[BN * (32 / 2 + 4)];




shared float16_t coopmat_stage[TM * TN * (BLOCK_SIZE / WARP)];


#line 1 "/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_id_funcs.glsl"










































































#line 138 "/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp"
#line 1 "/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl"
 void load_a_to_shmem(const uint pos_a, const uint row, const uint col, const uint idx_m, const uint block, const uint end_k) {


            const uint idx = pos_a + col * p.stride_a / 8 + row;
            const uint buf_idx = col * (32 / 2 + 4) + row * 8 / 2;
                       f16mat2x4 aa = f16mat2x4(data_a[idx]);
            buf_a[buf_idx] = aa[0].xy;
            buf_a[buf_idx + 1] = aa[0].zw;
            buf_a[buf_idx + 2] = aa[1].xy;
            buf_a[buf_idx + 3] = aa[1].zw;































































































































































































































































































































































































































































































































}


void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint idx_n, const uint block, const uint end_k) {


            const uint idx = pos_b + col * p.stride_b / 8 + row;
            const uint buf_idx = col * (32 / 2 + 4) + row * 8 / 2;
                       f16mat2x4 bb = f16mat2x4(data_b[idx]);
            buf_b[buf_idx + 0] = bb[0].xy;
            buf_b[buf_idx + 1] = bb[0].zw;
            buf_b[buf_idx + 2] = bb[1].xy;
            buf_b[buf_idx + 3] = bb[1].zw;






















}









































#line 139 "/home/taowen/projects/agentorch/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp"

void main() {
    const uint ic = gl_WorkGroupID.y;












    const uint batch_idx = gl_WorkGroupID.z + p.base_work_group_z;

    const uint i13 = batch_idx / p.ne12;
    const uint i12 = batch_idx % p.ne12;

    const uint i03 = i13 / p.broadcast3;
    const uint i02 = i12 / p.broadcast2;

    const uint batch_idx_a = i03 * p.ne02 + i02;


    const uint blocks_m = (p.M + BM - 1) / BM;
    const uint ir = gl_WorkGroupID.x % blocks_m;
    const uint ik = gl_WorkGroupID.x / blocks_m;

    const uint WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    const uint WSUBM = WM / WMITER;
    const uint WSUBN = WN / WNITER;


    const uint warp_i = gl_SubgroupID;

    const uint tiw = gl_SubgroupInvocationID;

    const uint cms_per_row = WM / TM;
    const uint cms_per_col = WN / TN;

    const uint storestride = WARP / TM;
    const uint store_r = tiw % TM;
    const uint store_c = tiw / TM;









    const uint warp_r = warp_i % (BM / WM);
    const uint warp_c = warp_i / (BM / WM);

    const uint loadr_a = gl_LocalInvocationID.x % (32 / 8 / 1);
    const uint loadc_a = gl_LocalInvocationID.x / (32 / 8 / 1);
    const uint loadr_b = gl_LocalInvocationID.x % (32 / 8 / 1);
    const uint loadc_b = gl_LocalInvocationID.x / (32 / 8 / 1);

    const uint loadstride_a = gl_WorkGroupSize.x * 8 * 1 / 32;
    const uint loadstride_b = gl_WorkGroupSize.x * 8 * 1 / 32;
































    const uint start_k = ik * p.k_split;
    const uint end_k = min(p.K, (ik + 1) * p.k_split);


    uint pos_a =



        batch_idx_a * (p.batch_stride_a / 8) +

        (ir * BM * p.stride_a + start_k) / 8;



    uint pos_b = (batch_idx * p.batch_stride_b + ic * BN * p.stride_b + start_k) / 8;



    coopmat < float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA > cache_a;
    coopmat < float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB > cache_b;
    coopmat < float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > sums[cms_per_row * cms_per_col];

    [[unroll]] for (uint i = 0; i < cms_per_row * cms_per_col; i ++) {
        sums[i] = coopmat < float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > (0.0f);
    }















    for (uint block = start_k; block < end_k; block += 32) {
        [[unroll]] for (uint l = 0; l < BM; l += loadstride_a) {
            load_a_to_shmem(pos_a, loadr_a, loadc_a + l, ir * BM + loadc_a + l, block, end_k);
        }
        [[unroll]] for (uint l = 0; l < BN; l += loadstride_b) {

            load_b_to_shmem(pos_b, loadr_b, loadc_b + l, ic * BN + loadc_b + l, block, end_k);



        }

        barrier();

        pos_a += 32 / 8;
        pos_b += 32 / 8;


        [[unroll]] for (uint i = 0; i < 32; i += TK) {
            [[unroll]] for (uint cm_row = 0; cm_row < cms_per_row; cm_row ++) {

                coopMatLoad(cache_a, buf_a, (warp_r * WM + cm_row * TM) * (32 / 2 + 4) + i / 2, (32 / 2 + 4), gl_CooperativeMatrixLayoutRowMajor);

                [[unroll]] for (uint cm_col = 0; cm_col < cms_per_col; cm_col ++) {
                    coopMatLoad(cache_b, buf_b, (warp_c * WN + cm_col * TN) * (32 / 2 + 4) + i / 2, (32 / 2 + 4), gl_CooperativeMatrixLayoutColumnMajor);

                    sums[cm_col * cms_per_row + cm_row] = coopMatMulAdd(cache_a, cache_b, sums[cm_col * cms_per_row + cm_row]);
                }
            }
        }












































        barrier();
    }



    [[unroll]] for (uint j = 0; j < cms_per_row * cms_per_col; j ++) {
        [[unroll]] for (uint i = 0; i < sums[j].length(); ++ i) {
            sums[j][i] = clamp(sums[j][i], - float16_t(65504.0), float16_t(65504.0));
        }
    }








    const uint dr = ir * BM + warp_r * WM;
    const uint dc = ic * BN + warp_c * WN;


    const uint offsets = batch_idx * p.batch_stride_d + ik * p.batch_stride_d * p.num_batches;























    const bool is_aligned = p.stride_d % 4 == 0;

    [[unroll]] for (uint cm_row = 0; cm_row < cms_per_row; cm_row ++) {
        [[unroll]] for (uint cm_col = 0; cm_col < cms_per_col; cm_col ++) {
            const bool is_in_bounds = dr + (cm_row + 1) * TM <= p.M && dc + (cm_col + 1) * TN <= p.N;

            if (is_aligned && is_in_bounds) {

                coopmat < float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > cm_dtype = coopmat < float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > (sums[cm_col * cms_per_row + cm_row]);
                coopMatStore(cm_dtype, data_d, offsets + (dc + cm_col * TN) * p.stride_d + dr + cm_row * TM, p.stride_d, gl_CooperativeMatrixLayoutColumnMajor);
            } else if (is_in_bounds) {

                coopMatStore(sums[cm_col * cms_per_row + cm_row], coopmat_stage, warp_i * TM * TN, TM, gl_CooperativeMatrixLayoutColumnMajor);

                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
                [[unroll]] for (uint col = 0; col < TN; col += storestride) {
                    data_d[offsets + (dc + cm_col * TN + col + store_c) * p.stride_d + dr + cm_row * TM + store_r] = float(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r]);
                }
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
            } else if (dr + cm_row * TM < p.M && dc + cm_col * TN < p.N) {

                coopMatStore(sums[cm_col * cms_per_row + cm_row], coopmat_stage, warp_i * TM * TN, TM, gl_CooperativeMatrixLayoutColumnMajor);

                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
                [[unroll]] for (uint col = 0; col < TN; col += storestride) {
                    if (dr + cm_row * TM + store_r < p.M && dc + cm_col * TN + col + store_c < p.N) {
                        data_d[offsets + (dc + cm_col * TN + col + store_c) * p.stride_d + dr + cm_row * TM + store_r] = float(coopmat_stage[warp_i * TM * TN + (col + store_c) * TM + store_r]);
                    }
                }
                controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
            }
        }
    }




































}
"""

_SPLIT_K_REDUCE_SOURCE = """\
#version 450

#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer A { float data_a[]; };
layout(binding = 0) readonly buffer A4 { vec4 data_a4[]; };
layout(binding = 1) writeonly buffer D { float data_d[]; };
layout(binding = 1) writeonly buffer D4 { vec4 data_d4[]; };

layout(push_constant) uniform parameter {
    uint ne;
    uint k_num;
} p;

void main() {

    const uint idx = gl_GlobalInvocationID.x * 4;

    if (idx >= p.ne) {
        return;
    }



    if (idx + 3 < p.ne && (p.ne % 4) == 0) {
        vec4 result = vec4(0.0f);

        [[unroll]] for (uint i = 0; i < p.k_num; i ++) {
            result += data_a4[ (i * p.ne + idx) / 4];
        }

        data_d4[idx / 4] = result;
    } else {
        [[unroll]] for (uint j = 0; j < 4; ++ j) {
            if (idx + j < p.ne) {
                float result = 0.0f;

                [[unroll]] for (uint i = 0; i < p.k_num; i ++) {
                    result += data_a[i * p.ne + idx + j];
                }

                data_d[idx + j] = result;
            }
        }
    }
}
"""

_MATMUL_SPECIALIZATION_CONSTANTS = {
    0: 256,
    1: 128,
    2: 128,
    3: 16,
    4: 64,
    5: 64,
    6: 2,
    7: 16,
    8: 16,
    9: 16,
    10: 64,
}

_MATMUL_EXECUTION_REQUIREMENTS = ShaderExecutionRequirements(
    cooperative_matrix=CooperativeMatrixRequirements(
        scope="subgroup",
        m_size=16,
        n_size=16,
        k_size=16,
        a_type="float16",
        b_type="float16",
        c_type="float16",
        result_type="float16",
    ),
    require_storage_buffer_16bit_access=True,
)

TensorBindings = Mapping[str, DeviceTensor]
ShapeSymbols = Mapping[str, int]


def qwen3_f16_prefill_split_k(*, out_features: int, steps: int, in_features: int) -> int:
    if in_features < 2048:
        return 1
    m_tiles = (out_features + 127) // 128
    n_tiles = (steps + 127) // 128
    tiles = max(1, m_tiles * n_tiles)
    split_k = min(96 // tiles, 8)
    if split_k < 1:
        split_k = 1
    while split_k > 1:
        chunk = _round_up(_ceil_div(in_features, split_k), 256)
        if chunk * (split_k - 1) < in_features:
            break
        split_k -= 1
    return split_k


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _round_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _shape(tensors: TensorBindings, name: str) -> tuple[int, ...]:
    return tensors[name].shape


def _output_shape(tensors: TensorBindings) -> tuple[int, ...]:
    output = tensors.get("output")
    if output is not None:
        return output.shape
    return tensors["split_k_output"].shape


def _matmul_k_split(tensors: TensorBindings, _shape_symbols: ShapeSymbols) -> int:
    weight_shape = _shape(tensors, "weight")
    out_shape = _output_shape(tensors)
    if len(out_shape) == 4:
        split_k = out_shape[0]
        in_features = weight_shape[1]
        return _round_up(_ceil_div(in_features, split_k), 256)
    return weight_shape[1]


def _matmul_num_batches(tensors: TensorBindings, _shape_symbols: ShapeSymbols) -> int:
    out_shape = _output_shape(tensors)
    return out_shape[1] if len(out_shape) == 4 else out_shape[0]


def _matmul_batch_stride_a(tensors: TensorBindings, _shape_symbols: ShapeSymbols) -> int:
    weight_shape = _shape(tensors, "weight")
    return weight_shape[0] * weight_shape[1]


def _matmul_batch_stride_b(tensors: TensorBindings, _shape_symbols: ShapeSymbols) -> int:
    x_shape = _shape(tensors, "x")
    return x_shape[1] * x_shape[2]


def _matmul_batch_stride_d(tensors: TensorBindings, _shape_symbols: ShapeSymbols) -> int:
    out_shape = _output_shape(tensors)
    return out_shape[2] * out_shape[3] if len(out_shape) == 4 else out_shape[1] * out_shape[2]


def _matmul_ne12(tensors: TensorBindings, _shape_symbols: ShapeSymbols) -> int:
    return _matmul_num_batches(tensors, _shape_symbols)


def _matmul_push_constants() -> PushConstantSpec:
    return push_constant_block(
        size=68,
        fields=(
            push_uint32(name="M", offset=0, value="N"),
            push_uint32(name="N", offset=4, value="S"),
            push_uint32(name="K", offset=8, value="K"),
            push_uint32(name="stride_a", offset=12, value="K"),
            push_uint32(name="stride_b", offset=16, value="K"),
            push_uint32(name="stride_d", offset=20, value="N"),
            push_uint32(name="batch_stride_a", offset=24, value=_matmul_batch_stride_a),
            push_uint32(name="batch_stride_b", offset=28, value=_matmul_batch_stride_b),
            push_uint32(name="batch_stride_d", offset=32, value=_matmul_batch_stride_d),
            push_uint32(name="base_work_group_z", offset=36, value=0),
            push_uint32(name="num_batches", offset=40, value=_matmul_num_batches),
            push_uint32(name="k_split", offset=44, value=_matmul_k_split),
            push_uint32(name="ne02", offset=48, value=1),
            push_uint32(name="ne12", offset=52, value=_matmul_ne12),
            push_uint32(name="broadcast2", offset=56, value=1),
            push_uint32(name="broadcast3", offset=60, value=1),
            push_uint32(name="padded_N", offset=64, value="S"),
        ),
    )


def _split_k_reduce_push_constants() -> PushConstantSpec:
    return push_constant_block(
        size=8,
        fields=(
            push_uint32(name="ne", offset=0, value=lambda tensors, _shape_symbols: tensors["output"].numel),
            push_uint32(name="k_num", offset=4, value=lambda tensors, _shape_symbols: tensors["split_k_input"].shape[0]),
        ),
    )


MATMUL_F16_F32_F16ACC_ALIGNED_CM1_SOURCE = _MATMUL_F16_F32_F16ACC_ALIGNED_CM1_SOURCE
MATMUL_EXECUTION_REQUIREMENTS = _MATMUL_EXECUTION_REQUIREMENTS
MATMUL_SPECIALIZATION_CONSTANTS = _MATMUL_SPECIALIZATION_CONSTANTS
matmul_push_constants = _matmul_push_constants


MATMUL_F16_F32_F16ACC_ALIGNED_L = shader_variant(
    name="matmul_f16_f32_f16acc_aligned_l",
    family="matmul_f16_f32_f16acc_aligned_l",
    execution_requirements=_MATMUL_EXECUTION_REQUIREMENTS,
    contract=shader_contract(
        class_name="MatmulF16F32F16AccAlignedLProgram",
        shader_name="matmul_f16_f32_f16acc_aligned_l",
        fields=(
            input_tensor(name="weight", binding="t_weight", role="weight", dtypes=("float16",), shape=("N", "K")),
            input_tensor(name="x", binding="t_x", role="x", dtypes=("float32",), shape=("B", "S", "K")),
            output_tensor(name="output", binding="t_output", role="output", dtypes=("float32",), shape=("B", "S", "N")),
        ),
        uniforms=(),
        push_constants=_matmul_push_constants(),
        dispatch=(ceil_div("N", 128), ceil_div("S", 128), "B"),
        bindings=(
            storage_buffer_binding(name="t_weight", binding=0),
            storage_buffer_binding(name="t_x", binding=1),
            storage_buffer_binding(name="t_output", binding=2),
        ),
    ),
    specialization_constants=_MATMUL_SPECIALIZATION_CONSTANTS,
    source=_MATMUL_F16_F32_F16ACC_ALIGNED_CM1_SOURCE,
)

MATMUL_F16_F32_F16ACC_ALIGNED_L_SPLIT = shader_variant(
    name="matmul_f16_f32_f16acc_aligned_l_split",
    family="matmul_f16_f32_f16acc_aligned_l",
    default_output_field="split_k_output",
    execution_requirements=_MATMUL_EXECUTION_REQUIREMENTS,
    contract=shader_contract(
        class_name="MatmulF16F32F16AccAlignedLSplitProgram",
        shader_name="matmul_f16_f32_f16acc_aligned_l_split",
        fields=(
            input_tensor(name="weight", binding="t_weight", role="weight", dtypes=("float16",), shape=("N", "K")),
            input_tensor(name="x", binding="t_x", role="x", dtypes=("float32",), shape=("B", "S", "K")),
            output_tensor(
                name="split_k_output",
                binding="t_split_k_output",
                role="split_k_output",
                dtypes=("float32",),
                shape=("SK", "B", "S", "N"),
            ),
        ),
        uniforms=(),
        push_constants=_matmul_push_constants(),
        dispatch=(mul(ceil_div("N", 128), "SK"), ceil_div("S", 128), "B"),
        bindings=(
            storage_buffer_binding(name="t_weight", binding=0),
            storage_buffer_binding(name="t_x", binding=1),
            storage_buffer_binding(name="t_split_k_output", binding=2),
        ),
    ),
    specialization_constants=_MATMUL_SPECIALIZATION_CONSTANTS,
    source=_MATMUL_F16_F32_F16ACC_ALIGNED_CM1_SOURCE,
)

SPLIT_K_REDUCE = shader_variant(
    name="split_k_reduce",
    family="split_k_reduce",
    contract=shader_contract(
        class_name="SplitKReduceProgram",
        shader_name="split_k_reduce",
        fields=(
            input_tensor(
                name="split_k_input",
                binding="t_split_k_input",
                role="split_k_input",
                dtypes=("float32",),
                shape=("SK", "B", "S", "N"),
            ),
            output_tensor(name="output", binding="t_output", role="output", dtypes=("float32",), shape=("B", "S", "N")),
        ),
        uniforms=(),
        push_constants=_split_k_reduce_push_constants(),
        dispatch=(ceil_div(mul(mul("B", "S"), "N"), 1024), 1, 1),
        bindings=(
            storage_buffer_binding(name="t_split_k_input", binding=0),
            storage_buffer_binding(name="t_output", binding=1),
        ),
    ),
    source=_SPLIT_K_REDUCE_SOURCE,
)
