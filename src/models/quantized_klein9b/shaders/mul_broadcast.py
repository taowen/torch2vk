"""Generated shader: mul_broadcast."""

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


MUL_BROADCAST = ShaderVariant(
    name='mul_broadcast',
    family='export',
    contract=ShaderContract(
        class_name='ExportMulBroadcastProgram',
        shader_name='mul_broadcast',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=(1, 1,)),
            ),
            TensorFieldSpec(
                name='y',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=(1, 'T',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=(1, 'T',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, 'T', dynamic=False),
                PushConstantFieldSpec('D0', PushConstantType.UINT32, 4, 1, dynamic=False),
                PushConstantFieldSpec('D1', PushConstantType.UINT32, 8, 'T', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div('T', 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float16_t y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint D0; uint D1; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) return;
    uint rem = idx;
    const uint c1 = rem % pc.D1;
    rem /= pc.D1;
    const uint c0 = rem % pc.D0;
    const uint x_idx = 0u;
    const uint y_idx = c1 * 1u;
    output_values[idx] = float16_t(x[x_idx] * y[y_idx]);
}
""",
)
