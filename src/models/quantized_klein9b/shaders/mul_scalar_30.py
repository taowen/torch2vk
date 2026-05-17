"""Generated shader: mul_scalar_30."""

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


MUL_SCALAR_30 = ShaderVariant(
    name='mul_scalar_30',
    family='export',
    contract=ShaderContract(
        class_name='ExportMulScalarProgram',
        shader_name='mul_scalar_30',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('B',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('B',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('scalar', PushConstantType.FLOAT32, 4, 1.0, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div('B', 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; float scalar; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = float(x[idx]) * pc.scalar; }
}
""",
)
