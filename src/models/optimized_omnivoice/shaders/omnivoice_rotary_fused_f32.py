"""Generated shader: omnivoice_rotary_fused_f32."""

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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements


OMNIVOICE_ROTARY_FUSED_F32 = ShaderVariant(
    name='omnivoice_rotary_fused_f32',
    family='optimized_omnivoice',
    contract=ShaderContract(
        class_name='OmniVoiceRotaryFusedF32Program',
        shader_name='omnivoice_rotary_fused_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('B', 'N', 'S', 'D',)),
            ),
            TensorFieldSpec(
                name='cos',
                io_kind=IOKind.INPUT,
                role='cos',
                contract=TensorContract(dtype='float16', shape=('B', 1, 'S', 'D',)),
            ),
            TensorFieldSpec(
                name='sin',
                io_kind=IOKind.INPUT,
                role='sin',
                contract=TensorContract(dtype='float16', shape=('B', 1, 'S', 'D',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('B', 'N', 'S', 'D',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('N', PushConstantType.UINT32, 4, 'N', dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 8, 'S', dynamic=False),
                PushConstantFieldSpec('D', PushConstantType.UINT32, 12, 'D', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul(mul('B', 'N'), 'S'), 'D'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly CosBuffer { float16_t cos_values[]; };
layout(set = 0, binding = 2) buffer restrict readonly SinBuffer { float16_t sin_values[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };

layout(push_constant) uniform PushConstants { uint B; uint N; uint S; uint D; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.N * pc.S * pc.D;
    if (idx >= total) {
        return;
    }

    const uint d = idx % pc.D;
    const uint seq_index = idx / pc.D;
    const uint s = seq_index % pc.S;
    const uint head_index = seq_index / pc.S;
    const uint b = head_index / pc.N;
    const uint half_d = pc.D >> 1u;
    const float x_value = float(x[idx]);
    const float rotated = (d < half_d) ? -float(x[idx + half_d]) : float(x[idx - half_d]);
    const uint rope_index = (b * pc.S + s) * pc.D + d;

    output_values[idx] = float16_t(fma(x_value, float(cos_values[rope_index]), rotated * float(sin_values[rope_index])));
}
""",
)
