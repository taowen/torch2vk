"""Generated shader: export_mean_dim_f32_8."""

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
)


EXPORT_MEAN_DIM_F32_8 = ShaderVariant(
    name='export_mean_dim_f32_8',
    family='export',
    contract=ShaderContract(
        class_name='ExportMeanDimProgram',
        shader_name='export_mean_dim_f32_8',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('S0', 'S1', 'S2', 'S3',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O0', 'O1', 'O2', 'O3',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('ROWS', PushConstantType.UINT32, 0, 2416, dynamic=False),
                PushConstantFieldSpec('COLS', PushConstantType.UINT32, 4, 128, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(2416, 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint ROWS; uint COLS; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float partial[256];
void main() {
    const uint row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    if (row >= pc.ROWS) { return; }
    float sum = 0.0;
    for (uint c = tid; c < pc.COLS; c += 256u) {
        sum += x[row * pc.COLS + c];
    }
    partial[tid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        barrier();
    }
    if (tid == 0u) { output_values[row] = partial[0] / float(pc.COLS); }
}
""",
)
