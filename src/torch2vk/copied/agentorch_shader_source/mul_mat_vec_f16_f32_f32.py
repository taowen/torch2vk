"""llama.cpp-shaped F16 x F32 mat-vec shader binding."""

from __future__ import annotations

from agentorch.kernel.contract import (
    ceil_div,
    input_tensor,
    inout_tensor,
    push_constant_input,
    shader_contract,
    storage_buffer_binding,
    tensor_binding_default,
)

from .llama_push_constants import mat_vec_push_constant_block
from .shader_execution_requirements import ShaderExecutionRequirements, SubgroupRequirements
from .shader_variant import ShaderBindingContext, shader_variant


_MUL_MAT_VEC_F16_SOURCE = """
#version 450


#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require


#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require









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





























layout(binding = 0) readonly buffer A { float16_t data_a[]; };










layout(binding = 1) readonly buffer B { float data_b[]; };

layout(binding = 1) readonly buffer BV2 { vec2 data_b_v2[]; };


layout(binding = 1) readonly buffer BV4 { vec4 data_b_v4[]; };


layout(binding = 2) writeonly buffer D { float data_d[]; };

layout(binding = 3) readonly buffer Fuse0 { float data_fuse0[]; };
layout(binding = 4) readonly buffer Fuse1 { float data_fuse1[]; };






layout(push_constant) uniform parameter
{
    uint ncols;
    uint stride_a;
    uint stride_b;
    uint stride_d;

    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;

    uint fusion_flags;







    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;

} p;





void get_offsets(out uint a_offset, out uint b_offset, out uint d_offset) {



    const uint batch_idx = gl_WorkGroupID.y + p.base_work_group_y;



    uint batch_idx_a = 0;
    if (batch_idx != 0) {
        const uint i13 = batch_idx / p.ne12;
        const uint i12 = batch_idx % p.ne12;

        const uint i03 = i13 / p.broadcast3;
        const uint i02 = i12 / p.broadcast2;

        batch_idx_a = i03 * p.ne02 + i02;
    }




    a_offset =



            batch_idx_a * (p.batch_stride_a / 1);

    b_offset =



            batch_idx * p.batch_stride_b;

    d_offset =



            batch_idx * p.batch_stride_d;

}

layout(constant_id = 0) const uint BLOCK_SIZE = 32;
layout(constant_id = 1) const uint NUM_ROWS = 1;
layout(constant_id = 2) const uint NUM_COLS = 1;


void reduce_result(inout float temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++ j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++ n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++ j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++ n) {













                if ( (p.fusion_flags & 0x1) != 0) {
                    temp[j][n] += float(data_fuse0[j * p.batch_stride_d + d_offset + first_row + n]);
                }
                if ( (p.fusion_flags & 0x2) != 0) {
                    temp[j][n] += float(data_fuse1[j * p.batch_stride_d + d_offset + first_row + n]);
                }
                data_d[j * p.batch_stride_d + d_offset + first_row + n] = float(temp[j][n]);
            }
        }
    }
}








































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(data_a[a_offset + ib], data_a[a_offset + ib + 1]);
}








































































































































































































































































































































































































































































vec2 get_dm(uint ib, uint a_offset) {
    return vec2(0, 0);
}



















































































































































































layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;








uint a_offset, b_offset, d_offset, y_offset;

void iter(inout float temp[NUM_COLS][NUM_ROWS], const uint first_row, const uint num_rows, const uint tid, const uint i, bool lastiter)
{
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++ j) {
        const uint col = i * BLOCK_SIZE + 2 * tid;
        const uint iqs = (col % 1) / 1;
        const uint iybs = col - col % 1;

















        const bool OOB = lastiter && (iybs + iqs + y_offset >= p.ncols);

                 float b0 = 0, b1 = 0;
        b0 = float(data_b[j * p.batch_stride_b + b_offset + iybs + iqs]);
        if (! OOB) {
            b1 = float(data_b[j * p.batch_stride_b + b_offset + iybs + iqs + y_offset]);
        }

        uint ibi = first_row * p.ncols;
        [[unroll]] for (uint n = 0; n < num_rows; ++ n) {
            const uint ib = (ibi + col) / 1;
            ibi += p.ncols;




















            const vec2 v = dequantize(ib, iqs, a_offset);


            temp[j][n] = fma(float(v.x), b0, temp[j][n]);
            if (! OOB) {
                temp[j][n] = fma(float(v.y), b1, temp[j][n]);
            }

        }
    }
}

void compute_outputs(const uint32_t first_row, const uint32_t num_rows) {
    const uint tid = gl_LocalInvocationID.x;

    get_offsets(a_offset, b_offset, d_offset);

    y_offset = 1 == 1 ? 1 : 1 / 2;

             float temp[NUM_COLS][NUM_ROWS];

    [[unroll]] for (uint j = 0; j < NUM_COLS; ++ j) {
        [[unroll]] for (uint i = 0; i < NUM_ROWS; ++ i) {
            temp[j][i] = float(0);
        }
    }

    uint num_iters = p.ncols / (2 * BLOCK_SIZE);
    if (num_iters * 2 * BLOCK_SIZE + 2 * tid < p.ncols) {
        num_iters ++;
    }
    int unroll_count = 4;
    uint unrolled_iters = num_iters & ~ (unroll_count - 1);




    if ( (p.ncols & 1) != 0 &&
        unrolled_iters == num_iters &&
        unrolled_iters > 0) {
        unrolled_iters -= unroll_count;
    }


    uint i = 0;
    while (i < unrolled_iters) {

        [[unroll]] for (uint k = 0; k < unroll_count; ++ k) {
            iter(temp, first_row, num_rows, tid, i * 2, false);
            i ++;
        }
    }

    unroll_count = 2;
    unrolled_iters = num_iters & ~ (unroll_count - 1);


    if ( (p.ncols & 1) != 0 &&
        unrolled_iters == num_iters &&
        unrolled_iters > 0) {
        unrolled_iters -= unroll_count;
    }


    while (i < unrolled_iters) {

        [[unroll]] for (uint k = 0; k < unroll_count; ++ k) {
            iter(temp, first_row, num_rows, tid, i * 2, false);
            i ++;
        }
    }
    while (i < num_iters) {
        iter(temp, first_row, num_rows, tid, i * 2, true);
        i ++;
    }

    reduce_result(temp, d_offset, first_row, num_rows, tid);
}

void main() {
    const uint first_row = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);






    if (first_row + NUM_ROWS <= p.stride_d) {
        compute_outputs(first_row, NUM_ROWS);
    } else {
        if (first_row >= p.stride_d) {
            return;
        }
        compute_outputs(first_row, p.stride_d - first_row);
    }
}
"""

_MUL_MAT_VEC_BF16_SOURCE = _MUL_MAT_VEC_F16_SOURCE.replace(
    "layout(binding = 0) readonly buffer A { float16_t data_a[]; };",
    "layout(binding = 0) readonly buffer A { uint16_t data_a[]; };",
).replace(
    "vec2 dequantize(uint ib, uint iqs, uint a_offset) {\n    return vec2(data_a[a_offset + ib], data_a[a_offset + ib + 1]);\n}",
    "vec2 dequantize(uint ib, uint iqs, uint a_offset) {\n"
    "    return vec2(\n"
    "        bf16_to_fp32(uint32_t(data_a[a_offset + ib])),\n"
    "        bf16_to_fp32(uint32_t(data_a[a_offset + ib + 1]))\n"
    "    );\n"
    "}",
)

_MUL_MAT_VEC_BF16_TORCH_PARITY_SOURCE = """
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer A { uint16_t data_a[]; };
layout(binding = 1) readonly buffer B { float data_b[]; };
layout(binding = 2) buffer D { float data_d[]; };
layout(binding = 3) readonly buffer F0 { float data_f0[]; };
layout(binding = 4) readonly buffer F1 { float data_f1[]; };

layout(push_constant) uniform PushConstants {
    uint ncols;
    uint stride_a;
    uint stride_b;
    uint stride_d;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;
    uint fusion_flags;
    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
} p;

float bf16_to_fp32(uint16_t bits) {
    return uintBitsToFloat(uint(bits) << 16);
}

void main() {
    uint row = gl_WorkGroupID.x;
    uint batch = gl_WorkGroupID.y;
    uint step = gl_WorkGroupID.z;

    uint nrows = p.stride_d;
    if (row >= nrows) {
        return;
    }

    uint a_base = row * p.stride_a;
    uint x_base = batch * p.batch_stride_b + step * p.stride_b;
    uint out_idx = batch * p.batch_stride_d + step * p.stride_d + row;

    float acc = 0.0;
    for (uint k = 0; k < p.ncols; ++k) {
        float w = bf16_to_fp32(data_a[a_base + k]);
        float x = data_b[x_base + k];
        acc += w * x;
    }

    if ((p.fusion_flags & 1u) != 0u) {
        acc += data_f0[out_idx];
    }
    data_d[out_idx] = acc;
}
"""


def _mat_vec_f16_specialization(context: ShaderBindingContext) -> dict[int, int]:
    steps = context.shape_symbols["S"]
    if steps < 1 or steps > 8:
        raise ValueError(f"mul_mat_vec_f16_f32_f32 expects 1 <= S <= 8, got {steps}")
    return {0: context.subgroup_size, 1: 2, 2: steps}


def _mat_vec_f16_execution_requirements(
    context: ShaderBindingContext,
) -> ShaderExecutionRequirements:
    return ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(
            required_size=context.subgroup_size, require_full_subgroups=True
        )
    )


def _mat_vec_bf16_specialization(context: ShaderBindingContext) -> dict[int, int]:
    steps = context.shape_symbols["S"]
    if steps < 1 or steps > 8:
        raise ValueError(f"mul_mat_vec_bf16_f32_f32 expects 1 <= S <= 8, got {steps}")
    return {0: context.subgroup_size, 1: 2, 2: steps}


MUL_MAT_VEC_F16_F32_F32 = shader_variant(
    name="mul_mat_vec_f16_f32_f32",
    family="mul_mat_vec_f16_f32_f32",
    contract=shader_contract(
        class_name="MulMatVecF16F32F32Program",
        shader_name="mul_mat_vec_f16_f32_f32",
        fields=(
            input_tensor(
                name="weight",
                binding="t_weight",
                role="weight",
                dtypes=("float16",),
                shape=("N", "K"),
            ),
            input_tensor(
                name="x", binding="t_x", role="x", dtypes=("float32",), shape=("B", "S", "K")
            ),
            inout_tensor(
                name="output",
                binding="t_output",
                role="output",
                dtypes=("float32",),
                shape=("B", "S", "N"),
            ),
            input_tensor(
                name="fuse0_placeholder",
                binding="t_fuse0_placeholder",
                role="fuse0_placeholder",
                dtypes=("float32",),
                shape=("B", "S", "N"),
            ),
            input_tensor(
                name="fuse1_placeholder",
                binding="t_fuse1_placeholder",
                role="fuse1_placeholder",
                dtypes=("float32",),
                shape=("B", "S", "N"),
            ),
        ),
        uniforms=(),
        push_constants=mat_vec_push_constant_block(
            batch="B",
            steps="S",
            in_features="K",
            out_features="N",
            fusion_flags=push_constant_input("fusion_flags"),
        ),
        dispatch=(ceil_div("N", 2), "B", 1),
        tensor_defaults=(
            tensor_binding_default(field_name="fuse0_placeholder", source_field_name="output"),
            tensor_binding_default(field_name="fuse1_placeholder", source_field_name="output"),
        ),
        bindings=(
            storage_buffer_binding(name="t_weight", binding=0),
            storage_buffer_binding(name="t_x", binding=1),
            storage_buffer_binding(name="t_output", binding=2),
            storage_buffer_binding(name="t_fuse0_placeholder", binding=3),
            storage_buffer_binding(name="t_fuse1_placeholder", binding=4),
        ),
    ),
    runtime_specialization_resolver=_mat_vec_f16_specialization,
    runtime_execution_requirements_resolver=_mat_vec_f16_execution_requirements,
    source=_MUL_MAT_VEC_F16_SOURCE,
)

MUL_MAT_VEC_BF16_F32_F32 = shader_variant(
    name="mul_mat_vec_bf16_f32_f32",
    family="mul_mat_vec_bf16_f32_f32",
    contract=shader_contract(
        class_name="MulMatVecBf16F32F32Program",
        shader_name="mul_mat_vec_bf16_f32_f32",
        fields=(
            input_tensor(
                name="weight",
                binding="t_weight",
                role="weight",
                dtypes=("bfloat16",),
                shape=("N", "K"),
            ),
            input_tensor(
                name="x", binding="t_x", role="x", dtypes=("float32",), shape=("B", "S", "K")
            ),
            inout_tensor(
                name="output",
                binding="t_output",
                role="output",
                dtypes=("float32",),
                shape=("B", "S", "N"),
            ),
            input_tensor(
                name="fuse0_placeholder",
                binding="t_fuse0_placeholder",
                role="fuse0_placeholder",
                dtypes=("float32",),
                shape=("B", "S", "N"),
            ),
            input_tensor(
                name="fuse1_placeholder",
                binding="t_fuse1_placeholder",
                role="fuse1_placeholder",
                dtypes=("float32",),
                shape=("B", "S", "N"),
            ),
        ),
        uniforms=(),
        push_constants=mat_vec_push_constant_block(
            batch="B",
            steps="S",
            in_features="K",
            out_features="N",
            fusion_flags=push_constant_input("fusion_flags"),
        ),
        dispatch=("N", "B", "S"),
        tensor_defaults=(
            tensor_binding_default(field_name="fuse0_placeholder", source_field_name="output"),
            tensor_binding_default(field_name="fuse1_placeholder", source_field_name="output"),
        ),
        bindings=(
            storage_buffer_binding(name="t_weight", binding=0),
            storage_buffer_binding(name="t_x", binding=1),
            storage_buffer_binding(name="t_output", binding=2),
            storage_buffer_binding(name="t_fuse0_placeholder", binding=3),
            storage_buffer_binding(name="t_fuse1_placeholder", binding=4),
        ),
    ),
    source=_MUL_MAT_VEC_BF16_TORCH_PARITY_SOURCE,
)
