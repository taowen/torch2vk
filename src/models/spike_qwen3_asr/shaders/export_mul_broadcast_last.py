"""Generated shader: export_mul_broadcast_last."""

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


EXPORT_MUL_BROADCAST_LAST = ShaderVariant(
    name='export_mul_broadcast_last',
    family='export',
    contract=ShaderContract(
        class_name='ExportMulBroadcastLastProgram',
        shader_name='export_mul_broadcast_last',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=(1, 'T', 'H',)),
            ),
            TensorFieldSpec(
                name='y',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=(1, 'T', 1,)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=(1, 'T', 'H',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul('T', 'H')),
                PushConstantFieldSpec('H', PushConstantType.UINT32, 4, 'H'),
            ),
        ),
        dispatch=(ceil_div(mul('T', 'H'), 256), 1, 1),
    ),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint H; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = x[idx] * y[idx / pc.H]; }
}
""",
)
