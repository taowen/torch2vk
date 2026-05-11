"""Generated shader: omnivoice_silu_mul_f32."""

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
    ceil_div,
    mul,
)


OMNIVOICE_SILU_MUL_F32 = ShaderVariant(
    name='omnivoice_silu_mul_f32',
    family='optimized_omnivoice',
    contract=ShaderContract(
        class_name='OmniVoiceSiluMulF32Program',
        shader_name='omnivoice_silu_mul_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('D0', 'D1', 'H',)),
            ),
            TensorFieldSpec(
                name='y',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('D0', 'D1', 'H',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('D0', 'D1', 'H',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul('D0', 'D1'), 'H'), dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('D0', 'D1'), 'H'), 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint N; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) {
        return;
    }
    const float value = x[idx];
    output_values[idx] = (value / (1.0 + exp(-value))) * y[idx];
}
""",
)
