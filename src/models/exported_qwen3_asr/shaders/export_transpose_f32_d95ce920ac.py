"""Generated shader: export_transpose_f32_d95ce920ac."""

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


EXPORT_TRANSPOSE_F32_D95CE920AC = ShaderVariant(
    name='export_transpose_f32_d95ce920ac',
    family='export',
    contract=ShaderContract(
        class_name='ExportTransposeProgram',
        shader_name='export_transpose_f32_d95ce920ac',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('I0', 'I1', 'I2',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O0', 'O1', 'O2',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, 1098240, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(1098240, 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        uint rem = idx;
        uint c2 = rem % 7680u;
        rem = rem / 7680u;
        uint c1 = rem % 13u;
        rem = rem / 13u;
        uint c0 = rem % 11u;
        rem = rem / 11u;
        uint in_idx = 0u;
        in_idx = in_idx * 11u + c0;
        in_idx = in_idx * 7680u + c2;
        in_idx = in_idx * 13u + c1;
        output_values[idx] = x[in_idx];
    }
}
""",
)
