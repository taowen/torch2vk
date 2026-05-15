"""Generated shader: select_int64_7."""

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


SELECT_INT64_7 = ShaderVariant(
    name='select_int64_7',
    family='export',
    contract=ShaderContract(
        class_name='ExportSelectProgram',
        shader_name='select_int64_7',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='int64', shape=('I0', 'I1', 'I2',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='int64', shape=('O0', 'O1',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul('O0', 'O1'), dynamic=False),
                PushConstantFieldSpec('select_dim', PushConstantType.UINT32, 4, 'I1', dynamic=False),
                PushConstantFieldSpec('inner', PushConstantType.UINT32, 8, 'I2', dynamic=False),
                PushConstantFieldSpec('selected', PushConstantType.UINT32, 12, 4, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul('O0', 'O1'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { int64_t x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { int64_t output_values[]; };
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
