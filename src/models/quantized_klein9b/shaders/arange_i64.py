"""Generated shader: arange_i64."""

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
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


ARANGE_I64 = ShaderVariant(
    name='arange_i64',
    family='export',
    contract=ShaderContract(
        class_name='ExportArangeI64Program',
        shader_name='arange_i64',
        fields=(
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='int64', shape=('B',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('start', PushConstantType.INT32, 4, 0, dynamic=False),
                PushConstantFieldSpec('step', PushConstantType.INT32, 8, 2, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div('B', 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { int64_t output_values[]; };
layout(push_constant) uniform PushConstants { uint N; int start; int step; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        output_values[idx] = int64_t(pc.start) + int64_t(idx) * int64_t(pc.step);
    }
}
""",
)
