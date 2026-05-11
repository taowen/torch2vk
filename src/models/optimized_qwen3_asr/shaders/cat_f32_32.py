"""Generated shader: cat_f32_32."""

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


CAT_F32_32 = ShaderVariant(
    name='cat_f32_32',
    family='export',
    contract=ShaderContract(
        class_name='ExportCatProgram',
        shader_name='cat_f32_32',
        fields=(
            TensorFieldSpec(
                name='a',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('A0', 'A1', 'A2', 'A3',)),
            ),
            TensorFieldSpec(
                name='b',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('B0', 'B1', 'B2', 'B3',)),
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
                PushConstantFieldSpec('N_OUT', PushConstantType.UINT32, 0, 154624, dynamic=False),
                PushConstantFieldSpec('A_STRIDE', PushConstantType.UINT32, 4, 64, dynamic=False),
                PushConstantFieldSpec('B_STRIDE', PushConstantType.UINT32, 8, 64, dynamic=False),
                PushConstantFieldSpec('OUT_STRIDE', PushConstantType.UINT32, 12, 128, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(154624, 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly ABuffer { float a[]; };
layout(set = 0, binding = 1) buffer restrict readonly BBuffer { float b[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N_OUT; uint A_STRIDE; uint B_STRIDE; uint OUT_STRIDE; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N_OUT) {
        uint row = idx / pc.OUT_STRIDE;
        uint col = idx % pc.OUT_STRIDE;
        if (col < pc.A_STRIDE) {
            output_values[idx] = a[row * pc.A_STRIDE + col];
        } else {
            output_values[idx] = b[row * pc.B_STRIDE + (col - pc.A_STRIDE)];
        }
    }
}
""",
)
