"""Generated shader: text_layer_export_add_f32."""

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


TEXT_LAYER_EXPORT_ADD_F32 = ShaderVariant(
    name='text_layer_export_add_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportAddF32Program',
        shader_name='text_layer_export_add_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=(1, 'T', 'H', 'D',)),
            ),
            TensorFieldSpec(
                name='y',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=(1, 'T', 'H', 'D',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=(1, 'T', 'H', 'D',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul('T', 'H'), 'D')),
            ),
        ),
        dispatch=(ceil_div(mul(mul('T', 'H'), 'D'), 256), 1, 1),
    ),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = x[idx] + y[idx]; }
}
""",
)
