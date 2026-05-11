"""Generated shader: layer_norm_bf16w_bf16b_f32."""

from __future__ import annotations

from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


LAYER_NORM_BF16W_BF16B_F32 = ShaderVariant(
    name='layer_norm_bf16w_bf16b_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportLayerNormBf16WeightBf16BiasProgram',
        shader_name='layer_norm_bf16w_bf16b_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('I0', 'I1',)),
            ),
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='bfloat16', shape=('W0',)),
            ),
            TensorFieldSpec(
                name='bias',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='bfloat16', shape=('W0',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('O0', 'O1',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec('ROWS', PushConstantType.UINT32, 0, 'I0', dynamic=False),
                PushConstantFieldSpec('COLS', PushConstantType.UINT32, 4, 'I1', dynamic=False),
                PushConstantFieldSpec('eps', PushConstantType.FLOAT32, 8, 1e-05, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=('I0', 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_bfloat16 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { bfloat16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { bfloat16_t bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint ROWS; uint COLS; float eps; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float partial_sum[256];
shared float partial_sumsq[256];
void main() {
    const uint row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    if (row >= pc.ROWS) { return; }
    float sum = 0.0;
    float sumsq = 0.0;
    for (uint c = tid; c < pc.COLS; c += 256u) {
        float v = float(x[row * pc.COLS + c]);
        sum += v;
        sumsq += v * v;
    }
    partial_sum[tid] = sum;
    partial_sumsq[tid] = sumsq;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        barrier();
    }
    float mean = partial_sum[0] / float(pc.COLS);
    float var = partial_sumsq[0] / float(pc.COLS) - mean * mean;
    float inv_std = inversesqrt(var + pc.eps);
    for (uint c = tid; c < pc.COLS; c += 256u) {
        uint idx = row * pc.COLS + c;
        output_values[idx] = float16_t((float(x[idx]) - mean) * inv_std * float(weight[c]) + float(bias[c]));
    }
}
""",
)
