"""Generated shader: slice_last_token_f16."""

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
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


SLICE_LAST_TOKEN_F16 = ShaderVariant(
    name='slice_last_token_f16',
    family='qwen3.text',
    contract=ShaderContract(
        class_name='Qwen3SliceLastTokenF16Program',
        shader_name='slice_last_token_f16',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('B', 'S', 'H',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('B', 1, 'H',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 4, 'S', dynamic=False),
                PushConstantFieldSpec('H', PushConstantType.UINT32, 8, 'H', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div('H', 256), 'B', 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 460

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float16_t out_values[]; };

layout(push_constant) uniform PushConstants { uint B; uint S; uint H; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint h = gl_GlobalInvocationID.x;
    const uint b = gl_GlobalInvocationID.y;
    if (b >= pc.B || h >= pc.H) {
        return;
    }
    out_values[b * pc.H + h] = x[(b * pc.S + (pc.S - 1u)) * pc.H + h];
}
""",
)
