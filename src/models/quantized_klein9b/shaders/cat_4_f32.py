"""Generated shader: cat_4_f32."""

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


CAT_4_F32 = ShaderVariant(
    name='cat_4_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportCat4Program',
        shader_name='cat_4_f32',
        fields=(
            TensorFieldSpec(
                name='x0',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('I0_0', 'I0_1', 'I0_2', 'I0_3', 'I0_4',)),
            ),
            TensorFieldSpec(
                name='x1',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('I1_0', 'I1_1', 'I1_2', 'I1_3', 'I1_4',)),
            ),
            TensorFieldSpec(
                name='x2',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('I2_0', 'I2_1', 'I2_2', 'I2_3', 'I2_4',)),
            ),
            TensorFieldSpec(
                name='x3',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('I3_0', 'I3_1', 'I3_2', 'I3_3', 'I3_4',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('O0', 'O1', 'O2', 'O3', 'O4',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('N_OUT', PushConstantType.UINT32, 0, mul(mul(mul(mul('O0', 'O1'), 'O2'), 'O3'), 'O4'), dynamic=False),
                PushConstantFieldSpec('OUT_STRIDE', PushConstantType.UINT32, 4, 256, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul(mul(mul('O0', 'O1'), 'O2'), 'O3'), 'O4'), 256), 1, 1),
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
layout(push_constant) uniform PushConstants { uint N_OUT; uint OUT_STRIDE; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N_OUT) {
        uint row = idx / pc.OUT_STRIDE;
        uint col = idx % pc.OUT_STRIDE;
        if (col < 64u) {
            output_values[idx] = x0[row * 64u + (col - 0u)];
        }
        else if (col < 128u) {
            output_values[idx] = x1[row * 64u + (col - 64u)];
        }
        else if (col < 192u) {
            output_values[idx] = x2[row * 64u + (col - 128u)];
        }
        else if (col < 256u) {
            output_values[idx] = x3[row * 64u + (col - 192u)];
        }
    }
}
""",
)
