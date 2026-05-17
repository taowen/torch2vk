"""Generated shader: mul_broadcast_852."""

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


MUL_BROADCAST_852 = ShaderVariant(
    name='mul_broadcast_852',
    family='export',
    contract=ShaderContract(
        class_name='ExportMulBroadcastProgram',
        shader_name='mul_broadcast_852',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=(1, 1, 'D2', 'D3', 'D4',)),
            ),
            TensorFieldSpec(
                name='y',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=(1, 'D1', 'D2', 'D3', 1,)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=(1, 'D1', 'D2', 'D3', 'D4',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=24,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul(mul('D1', 'D2'), 'D3'), 'D4'), dynamic=False),
                PushConstantFieldSpec('D0', PushConstantType.UINT32, 4, 1, dynamic=False),
                PushConstantFieldSpec('D1', PushConstantType.UINT32, 8, 'D1', dynamic=False),
                PushConstantFieldSpec('D2', PushConstantType.UINT32, 12, 'D2', dynamic=False),
                PushConstantFieldSpec('D3', PushConstantType.UINT32, 16, 'D3', dynamic=False),
                PushConstantFieldSpec('D4', PushConstantType.UINT32, 20, 'D4', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul(mul('D1', 'D2'), 'D3'), 'D4'), 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint D0; uint D1; uint D2; uint D3; uint D4; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) return;
    uint rem = idx;
    const uint c4 = rem % pc.D4;
    rem /= pc.D4;
    const uint c3 = rem % pc.D3;
    rem /= pc.D3;
    const uint c2 = rem % pc.D2;
    rem /= pc.D2;
    const uint c1 = rem % pc.D1;
    rem /= pc.D1;
    const uint c0 = rem % pc.D0;
    const uint x_idx = c2 * pc.D3 * pc.D4 + c3 * pc.D4 + c4 * 1u;
    const uint y_idx = c1 * pc.D2 * pc.D3 * 1u + c2 * pc.D3 * 1u + c3 * 1u;
    output_values[idx] = x[x_idx] * y[y_idx];
}
""",
)
