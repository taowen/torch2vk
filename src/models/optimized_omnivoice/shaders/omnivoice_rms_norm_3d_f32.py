"""Generated shader: omnivoice_rms_norm_3d_f32."""

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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements


OMNIVOICE_RMS_NORM_3D_F32 = ShaderVariant(
    name='omnivoice_rms_norm_3d_f32',
    family='optimized_omnivoice',
    contract=ShaderContract(
        class_name='OmniVoiceRmsNorm3DF32Program',
        shader_name='omnivoice_rms_norm_3d_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('D0', 'D1', 'H',)),
            ),
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='float32', shape=('H',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('D0', 'D1', 'H',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('rows', PushConstantType.UINT32, 0, mul('D0', 'D1'), dynamic=False),
                PushConstantFieldSpec('H', PushConstantType.UINT32, 4, 'H', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(mul('D0', 'D1'), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };

layout(push_constant) uniform PushConstants { uint rows; uint H; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared float partial[256];

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint lane = gl_LocalInvocationID.x;
    float sum = 0.0;

    for (uint h = lane; h < pc.H; h += 256u) {
        const float value = float(x[row * pc.H + h]);
        sum += value * value;
    }
    partial[lane] = sum;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        barrier();
    }

    const float scale = inversesqrt(partial[0] / float(pc.H) + 0.000001);
    for (uint h = lane; h < pc.H; h += 256u) {
        const uint index = row * pc.H + h;
        output_values[index] = float16_t(float(x[index]) * scale * weight[h]);
    }
}
""",
)
