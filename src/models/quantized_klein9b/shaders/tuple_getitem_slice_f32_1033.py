"""Generated shader: tuple_getitem_slice_f32_1033."""

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


TUPLE_GETITEM_SLICE_F32_1033 = ShaderVariant(
    name='tuple_getitem_slice_f32_1033',
    family='export',
    contract=ShaderContract(
        class_name='ExportTupleGetitemSliceProgram',
        shader_name='tuple_getitem_slice_f32_1033',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('I0', 'I1', 'I2',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('O0', 'O1', 'O2',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec('N_OUT', PushConstantType.UINT32, 0, mul(mul('O0', 'O1'), 'O2'), dynamic=False),
                PushConstantFieldSpec('IN_STRIDE', PushConstantType.UINT32, 4, 24576, dynamic=False),
                PushConstantFieldSpec('OUT_STRIDE', PushConstantType.UINT32, 8, 12288, dynamic=False),
                PushConstantFieldSpec('OFFSET', PushConstantType.UINT32, 12, 12288, dynamic=False),
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
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint N_OUT; uint IN_STRIDE; uint OUT_STRIDE; uint OFFSET; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N_OUT) {
        uint row = idx / pc.OUT_STRIDE;
        uint col = idx % pc.OUT_STRIDE;
        output_values[idx] = x[row * pc.IN_STRIDE + pc.OFFSET + col];
    }
}
""",
)
