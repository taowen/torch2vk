"""Generated shader: transpose_f32_322f87bdab."""

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


TRANSPOSE_F32_322F87BDAB = ShaderVariant(
    name='transpose_f32_322f87bdab',
    family='export',
    contract=ShaderContract(
        class_name='ExportTransposeProgram',
        shader_name='transpose_f32_322f87bdab',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('I0', 'I1', 'I2', 'I3',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O0', 'O1', 'O2', 'O3',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, 348160, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(348160, 256), 1, 1),
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
        uint c3 = rem % 128u;
        rem = rem / 128u;
        uint c2 = rem % 85u;
        rem = rem / 85u;
        uint c1 = rem % 16u;
        rem = rem / 16u;
        uint c0 = rem % 2u;
        rem = rem / 2u;
        uint in_idx = 0u;
        in_idx = in_idx * 2u + c0;
        in_idx = in_idx * 85u + c2;
        in_idx = in_idx * 16u + c1;
        in_idx = in_idx * 128u + c3;
        output_values[idx] = x[in_idx];
    }
}
""",
)
