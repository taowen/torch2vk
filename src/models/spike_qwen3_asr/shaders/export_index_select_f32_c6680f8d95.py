"""Generated shader: export_index_select_f32_c6680f8d95."""

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


EXPORT_INDEX_SELECT_F32_C6680F8D95 = ShaderVariant(
    name='export_index_select_f32_c6680f8d95',
    family='export',
    contract=ShaderContract(
        class_name='ExportIndexSelectF32Program',
        shader_name='export_index_select_f32_c6680f8d95',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('N', 'H',)),
            ),
            TensorFieldSpec(
                name='index',
                io_kind=IOKind.INPUT,
                role='index',
                contract=TensorContract(dtype='int64', shape=('O',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O', 'H',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('O', PushConstantType.UINT32, 0, 'O', dynamic=False),
                PushConstantFieldSpec('H', PushConstantType.UINT32, 4, 'H', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul('O', 'H'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndexBuffer { int64_t index_values[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint O; uint H; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.O * pc.H;
    if (idx >= total) { return; }

    const uint row = idx / pc.H;
    const uint h = idx - row * pc.H;
    const uint src_row = uint(index_values[row]);
    output_values[idx] = x[src_row * pc.H + h];
}
""",
)
