"""Generated shader: export_max_f32."""

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


EXPORT_MAX_F32 = ShaderVariant(
    name='export_max_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportMaxProgram',
        shader_name='export_max_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('I0',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=(1,)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, 2),
            ),
        ),
        dispatch=(1, 1, 1),
    ),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float partial_max[256];
void main() {
    const uint tid = gl_LocalInvocationID.x;
    float local_max = -1.0e38;
    for (uint i = tid; i < pc.N; i += 256u) {
        local_max = max(local_max, x[i]);
    }
    partial_max[tid] = local_max;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial_max[tid] = max(partial_max[tid], partial_max[tid + stride]); }
        barrier();
    }
    if (tid == 0u) { output_values[0] = partial_max[0]; }
}
""",
)
