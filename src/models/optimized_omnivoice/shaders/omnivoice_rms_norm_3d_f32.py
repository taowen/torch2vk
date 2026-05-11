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
                contract=TensorContract(dtype='float32', shape=('D0', 'D1', 'H',)),
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
                contract=TensorContract(dtype='float32', shape=('D0', 'D1', 'H',)),
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
    execution_requirements=None,
    source="""\
#version 450

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint rows; uint H; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared float partial[256];

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint lane = gl_LocalInvocationID.x;
    float sum = 0.0;

    for (uint h = lane; h < pc.H; h += 256u) {
        const float value = x[row * pc.H + h];
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
        output_values[index] = x[index] * scale * weight[h];
    }
}
""",
)
