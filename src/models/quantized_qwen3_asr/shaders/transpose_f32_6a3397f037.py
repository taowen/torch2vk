"""Generated shader: transpose_f32_6a3397f037."""

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


TRANSPOSE_F32_6A3397F037 = ShaderVariant(
    name='transpose_f32_6a3397f037',
    family='export',
    contract=ShaderContract(
        class_name='ExportTransposeProgram',
        shader_name='transpose_f32_6a3397f037',
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
            size=28,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul('O0', 'O1'), 'O2'), dynamic=False),
                PushConstantFieldSpec('O0', PushConstantType.UINT32, 4, 'O0', dynamic=False),
                PushConstantFieldSpec('O1', PushConstantType.UINT32, 8, 'O1', dynamic=False),
                PushConstantFieldSpec('O2', PushConstantType.UINT32, 12, 'O2', dynamic=False),
                PushConstantFieldSpec('I0', PushConstantType.UINT32, 16, 'I0', dynamic=False),
                PushConstantFieldSpec('I1', PushConstantType.UINT32, 20, 'I1', dynamic=False),
                PushConstantFieldSpec('I2', PushConstantType.UINT32, 24, 'I2', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('O0', 'O1'), 'O2'), 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants {
    uint N;
    uint O0;
    uint O1;
    uint O2;
    uint I0;
    uint I1;
    uint I2;
} pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        uint rem = idx;
        uint c2 = rem % pc.O2;
        rem = rem / pc.O2;
        uint c1 = rem % pc.O1;
        rem = rem / pc.O1;
        uint c0 = rem % pc.O0;
        rem = rem / pc.O0;
        uint in_idx = 0u;
        in_idx = in_idx * pc.I0 + c1;
        in_idx = in_idx * pc.I1 + c0;
        in_idx = in_idx * pc.I2 + c2;
        output_values[idx] = x[in_idx];
    }
}
""",
)
