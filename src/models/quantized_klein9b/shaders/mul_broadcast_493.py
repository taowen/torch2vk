"""Generated shader: mul_broadcast_493."""

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
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


MUL_BROADCAST_493 = ShaderVariant(
    name='mul_broadcast_493',
    family='export',
    contract=ShaderContract(
        class_name='ExportMulBroadcastProgram',
        shader_name='mul_broadcast_493',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=(1, 1, 'H',)),
            ),
            TensorFieldSpec(
                name='y',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=(1, 'T', 'H',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=(1, 'T', 'H',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul('T', 'H'), dynamic=False),
                PushConstantFieldSpec('D0', PushConstantType.UINT32, 4, 1, dynamic=False),
                PushConstantFieldSpec('D1', PushConstantType.UINT32, 8, 'T', dynamic=False),
                PushConstantFieldSpec('D2', PushConstantType.UINT32, 12, 'H', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul('T', 'H'), 256), 1, 1),
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
layout(push_constant) uniform PushConstants { uint N; uint D0; uint D1; uint D2; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) return;
    uint rem = idx;
    const uint c2 = rem % pc.D2;
    rem /= pc.D2;
    const uint c1 = rem % pc.D1;
    rem /= pc.D1;
    const uint c0 = rem % pc.D0;
    const uint x_idx = c2 * 1u;
    const uint y_idx = c1 * pc.D2 + c2 * 1u;
    output_values[idx] = float16_t(x[x_idx] * y[y_idx]);
}
""",
)
