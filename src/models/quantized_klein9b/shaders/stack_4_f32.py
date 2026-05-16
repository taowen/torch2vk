"""Generated shader: stack_4_f32."""

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


STACK_4_F32 = ShaderVariant(
    name='stack_4_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportStack4Program',
        shader_name='stack_4_f32',
        fields=(
            TensorFieldSpec(
                name='x0',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=(1, 'T', 'H',)),
            ),
            TensorFieldSpec(
                name='x1',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=(1, 'T', 'H',)),
            ),
            TensorFieldSpec(
                name='x2',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=(1, 'T', 'H',)),
            ),
            TensorFieldSpec(
                name='x3',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=(1, 'T', 'H',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=(1, 'T', 'H', 'D',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul('T', 'H'), 'D'), dynamic=False),
                PushConstantFieldSpec('INNER', PushConstantType.UINT32, 4, 1, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('T', 'H'), 'D'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly X0Buffer { float16_t x0[]; };
layout(set = 0, binding = 1) buffer restrict readonly X1Buffer { float16_t x1[]; };
layout(set = 0, binding = 2) buffer restrict readonly X2Buffer { float16_t x2[]; };
layout(set = 0, binding = 3) buffer restrict readonly X3Buffer { float16_t x3[]; };
layout(set = 0, binding = 4) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint INNER; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const uint source = (idx / pc.INNER) % 4u;
    const uint source_idx = (idx / (pc.INNER * 4u)) * pc.INNER + (idx % pc.INNER);
    if (source == 0u) { output_values[idx] = x0[source_idx]; }
    else if (source == 1u) { output_values[idx] = x1[source_idx]; }
    else if (source == 2u) { output_values[idx] = x2[source_idx]; }
    else if (source == 3u) { output_values[idx] = x3[source_idx]; }
}
""",
)
