"""Generated shader: mul_broadcast_inner_59."""

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
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


MUL_BROADCAST_INNER_59 = ShaderVariant(
    name='mul_broadcast_inner_59',
    family='export',
    contract=ShaderContract(
        class_name='ExportMulBroadcastInnerProgram',
        shader_name='mul_broadcast_inner_59',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=(1, 'T', 1,)),
            ),
            TensorFieldSpec(
                name='y',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=(1, 'T', 'H',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=(1, 'T', 'H',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul('T', 'H'), dynamic=False),
                PushConstantFieldSpec('STRIDE', PushConstantType.UINT32, 4, 1, dynamic=False),
                PushConstantFieldSpec('REPEAT', PushConstantType.UINT32, 8, 'H', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul('T', 'H'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint STRIDE; uint REPEAT; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        uint broadcast_idx = (idx / pc.STRIDE) / pc.REPEAT * pc.STRIDE + idx % pc.STRIDE;
        output_values[idx] = float16_t(x[broadcast_idx] * y[idx]);
    }
}
""",
)
