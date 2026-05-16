"""Generated shader: group_norm_f32w_f32b_f32."""

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
    mul,
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


GROUP_NORM_F32W_F32B_F32 = ShaderVariant(
    name='group_norm_f32w_f32b_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportGroupNormF32WeightF32BiasProgram',
        shader_name='group_norm_f32w_f32b_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('I0', 'I1', 'I2', 'I3',)),
            ),
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='float32', shape=('C',)),
            ),
            TensorFieldSpec(
                name='bias',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('C',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('O0', 'O1', 'O2', 'O3',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=24,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 'I0', dynamic=False),
                PushConstantFieldSpec('C', PushConstantType.UINT32, 4, 'I1', dynamic=False),
                PushConstantFieldSpec('H', PushConstantType.UINT32, 8, 'I2', dynamic=False),
                PushConstantFieldSpec('W', PushConstantType.UINT32, 12, 'I3', dynamic=False),
                PushConstantFieldSpec('G', PushConstantType.UINT32, 16, 32, dynamic=False),
                PushConstantFieldSpec('eps', PushConstantType.FLOAT32, 20, 1e-06, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(mul('I0', 32), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { float bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint C; uint H; uint W; uint G; float eps; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float partial_sum[256];
shared float partial_sumsq[256];
void main() {
    const uint group_row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    const uint channels_per_group = pc.C / pc.G;
    const uint cols = channels_per_group * pc.H * pc.W;
    const uint b = group_row / pc.G;
    const uint g = group_row % pc.G;
    if (b >= pc.B) { return; }

    float sum = 0.0;
    float sumsq = 0.0;
    for (uint col = tid; col < cols; col += 256u) {
        const uint hw = col % (pc.H * pc.W);
        const uint channel_in_group = col / (pc.H * pc.W);
        const uint channel = g * channels_per_group + channel_in_group;
        const uint idx = ((b * pc.C + channel) * pc.H * pc.W) + hw;
        const float v = float(x[idx]);
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
    const float mean = partial_sum[0] / float(cols);
    const float var = partial_sumsq[0] / float(cols) - mean * mean;
    const float inv_std = inversesqrt(var + pc.eps);

    for (uint col = tid; col < cols; col += 256u) {
        const uint hw = col % (pc.H * pc.W);
        const uint channel_in_group = col / (pc.H * pc.W);
        const uint channel = g * channels_per_group + channel_in_group;
        const uint idx = ((b * pc.C + channel) * pc.H * pc.W) + hw;
        output_values[idx] = float16_t((float(x[idx]) - mean) * inv_std * float(weight[channel]) + float(bias[channel]));
    }
}
""",
)
