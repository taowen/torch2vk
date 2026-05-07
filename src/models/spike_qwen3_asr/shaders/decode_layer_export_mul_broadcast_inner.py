"""Generated shader: decode_layer_export_mul_broadcast_inner."""

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


DECODE_LAYER_EXPORT_MUL_BROADCAST_INNER = ShaderVariant(
    name='decode_layer_export_mul_broadcast_inner',
    family='export',
    contract=ShaderContract(
        class_name='ExportMulBroadcastInnerProgram',
        shader_name='decode_layer_export_mul_broadcast_inner',
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
                contract=TensorContract(dtype='float32', shape=(1, 1, 1, 'D',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=(1, 'T', 'H', 'D',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul('T', 'H'), 'D'), dynamic=False),
                PushConstantFieldSpec('STRIDE', PushConstantType.UINT32, 4, 128, dynamic=False),
                PushConstantFieldSpec('REPEAT', PushConstantType.UINT32, 8, 16, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('T', 'H'), 'D'), 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint STRIDE; uint REPEAT; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        uint y_idx = (idx / pc.STRIDE) / pc.REPEAT * pc.STRIDE + idx % pc.STRIDE;
        output_values[idx] = x[idx] * y[y_idx];
    }
}
""",
)
