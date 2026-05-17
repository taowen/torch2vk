"""Generated shader: select_float32_981."""

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


SELECT_FLOAT32_981 = ShaderVariant(
    name='select_float32_981',
    family='export',
    contract=ShaderContract(
        class_name='ExportSelectProgram',
        shader_name='select_float32_981',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('I0', 'I1', 'I2', 'I3', 'I4', 'I5',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O0', 'O1', 'O2', 'O3', 'O4',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul(mul(mul('O0', 'O1'), 'O2'), 'O3'), 'O4'), dynamic=False),
                PushConstantFieldSpec('select_dim', PushConstantType.UINT32, 4, 'I5', dynamic=False),
                PushConstantFieldSpec('inner', PushConstantType.UINT32, 8, 1, dynamic=False),
                PushConstantFieldSpec('selected', PushConstantType.UINT32, 12, 1, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul(mul(mul('O0', 'O1'), 'O2'), 'O3'), 'O4'), 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint select_dim; uint inner; uint selected; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) return;
    const uint outer = idx / pc.inner;
    const uint inner = idx - outer * pc.inner;
    output_values[idx] = x[(outer * pc.select_dim + pc.selected) * pc.inner + inner];
}
""",
)
