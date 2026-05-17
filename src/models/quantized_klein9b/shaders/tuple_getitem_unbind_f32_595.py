"""Generated shader: tuple_getitem_unbind_f32_595."""

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


TUPLE_GETITEM_UNBIND_F32_595 = ShaderVariant(
    name='tuple_getitem_unbind_f32_595',
    family='export',
    contract=ShaderContract(
        class_name='ExportTupleGetitemUnbindProgram',
        shader_name='tuple_getitem_unbind_f32_595',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('I0', 'I1', 'I2', 'I3', 'I4',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O0', 'O1', 'O2', 'O3',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul(mul('O0', 'O1'), 'O2'), 'O3'), dynamic=False),
                PushConstantFieldSpec('SELECT_DIM', PushConstantType.UINT32, 4, 'I0', dynamic=False),
                PushConstantFieldSpec('INNER', PushConstantType.UINT32, 8, mul(mul(mul('I1', 'I2'), 'I3'), 'I4'), dynamic=False),
                PushConstantFieldSpec('SELECTED', PushConstantType.UINT32, 12, 1, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul(mul('O0', 'O1'), 'O2'), 'O3'), 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint SELECT_DIM; uint INNER; uint SELECTED; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const uint outer = idx / pc.INNER;
    const uint inner = idx - outer * pc.INNER;
    output_values[idx] = x[(outer * pc.SELECT_DIM + pc.SELECTED) * pc.INNER + inner];
}
""",
)
