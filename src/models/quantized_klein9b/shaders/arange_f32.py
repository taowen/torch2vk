"""Generated shader: arange_f32."""

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


ARANGE_F32 = ShaderVariant(
    name='arange_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportArangeProgram',
        shader_name='arange_f32',
        fields=(
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('B',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('start', PushConstantType.FLOAT32, 4, 0.0, dynamic=False),
                PushConstantFieldSpec('step', PushConstantType.FLOAT32, 8, 1.0, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div('B', 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; float start; float step; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = pc.start + float(idx) * pc.step; }
}
""",
)
