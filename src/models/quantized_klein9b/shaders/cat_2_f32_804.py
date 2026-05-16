"""Generated shader: cat_2_f32_804."""

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


CAT_2_F32_804 = ShaderVariant(
    name='cat_2_f32_804',
    family='export',
    contract=ShaderContract(
        class_name='ExportCat2Program',
        shader_name='cat_2_f32_804',
        fields=(
            TensorFieldSpec(
                name='x0',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('I0_0', 'I0_1', 'I0_2',)),
            ),
            TensorFieldSpec(
                name='x1',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('I1_0', 'I1_1', 'I1_2',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('O0', 'O1', 'O2',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('N_OUT', PushConstantType.UINT32, 0, mul(mul('O0', 'O1'), 'O2'), dynamic=False),
                PushConstantFieldSpec('OUT_STRIDE', PushConstantType.UINT32, 4, 16384, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('O0', 'O1'), 'O2'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly X0Buffer { float16_t x0[]; };
layout(set = 0, binding = 1) buffer restrict readonly X1Buffer { float16_t x1[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint N_OUT; uint OUT_STRIDE; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N_OUT) {
        uint row = idx / pc.OUT_STRIDE;
        uint col = idx % pc.OUT_STRIDE;
        if (col < 4096u) {
            output_values[idx] = x0[row * 4096u + (col - 0u)];
        }
        else if (col < 16384u) {
            output_values[idx] = x1[row * 12288u + (col - 4096u)];
        }
    }
}
""",
)
