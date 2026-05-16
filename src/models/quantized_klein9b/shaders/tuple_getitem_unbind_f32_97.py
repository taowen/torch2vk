"""Generated shader: tuple_getitem_unbind_f32_97."""

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


TUPLE_GETITEM_UNBIND_F32_97 = ShaderVariant(
    name='tuple_getitem_unbind_f32_97',
    family='export',
    contract=ShaderContract(
        class_name='ExportTupleGetitemUnbindProgram',
        shader_name='tuple_getitem_unbind_f32_97',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('I0', 'I1', 'I2', 'I3', 'I4',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('O0', 'O1', 'O2', 'O3',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul(mul('O0', 'O1'), 'O2'), 'O3'), dynamic=False),
                PushConstantFieldSpec('SELECT_DIM', PushConstantType.UINT32, 4, 'I0', dynamic=False),
                PushConstantFieldSpec('INNER', PushConstantType.UINT32, 8, 4194304, dynamic=False),
                PushConstantFieldSpec('SELECTED', PushConstantType.UINT32, 12, 2, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul(mul('O0', 'O1'), 'O2'), 'O3'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
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
