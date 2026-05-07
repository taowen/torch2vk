"""Generated shader: decode_layer_export_transpose_f32."""

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


DECODE_LAYER_EXPORT_TRANSPOSE_F32 = ShaderVariant(
    name='decode_layer_export_transpose_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportTransposeProgram',
        shader_name='decode_layer_export_transpose_f32',
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
            size=16,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, 2048),
                PushConstantFieldSpec('D1', PushConstantType.UINT32, 4, 1),
                PushConstantFieldSpec('D2', PushConstantType.UINT32, 8, 16),
                PushConstantFieldSpec('D3', PushConstantType.UINT32, 12, 128),
            ),
        ),
        dispatch=(ceil_div(2048, 256), 1, 1),
    ),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint D1; uint D2; uint D3; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        // output layout: (B, D2, D1, D3) — transposed dims 1 and 2
        uint d3 = idx % pc.D3;
        uint rem = idx / pc.D3;
        uint d1 = rem % pc.D1;
        rem = rem / pc.D1;
        uint d2 = rem % pc.D2;
        uint b = rem / pc.D2;
        // input layout: (B, D1, D2, D3)
        uint in_idx = ((b * pc.D1 + d1) * pc.D2 + d2) * pc.D3 + d3;
        output_values[idx] = x[in_idx];
    }
}
""",
)
