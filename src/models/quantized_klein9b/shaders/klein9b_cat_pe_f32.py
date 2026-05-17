"""Generated shader: klein9b_cat_pe_f32."""

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
    add,
    ceil_div,
    mul,
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


KLEIN9B_CAT_PE_F32 = ShaderVariant(
    name='klein9b_cat_pe_f32',
    family='klein9b',
    contract=ShaderContract(
        class_name='Klein9BCatPeF32Program',
        shader_name='klein9b_cat_pe_f32',
        fields=(
            TensorFieldSpec(
                name='pe_ctx',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('B', 'A', 'T', 'D', 'R', 'C',)),
            ),
            TensorFieldSpec(
                name='pe_x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('B', 'A', 'S', 'D', 'R', 'C',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('B', 'A', add('T', 'S'), 'D', 'R', 'C',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul(mul(mul(mul('B', 'A'), add('T', 'S')), 'D'), 'R'), 'C'), dynamic=False),
                PushConstantFieldSpec('T', PushConstantType.UINT32, 4, 'T', dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 8, 'S', dynamic=False),
                PushConstantFieldSpec('INNER', PushConstantType.UINT32, 12, mul(mul('D', 'R'), 'C'), dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul(mul(mul(mul('B', 'A'), add('T', 'S')), 'D'), 'R'), 'C'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly PeCtxBuffer { float pe_ctx[]; };
layout(set = 0, binding = 1) buffer restrict readonly PeXBuffer { float pe_x[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint T; uint S; uint INNER; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const uint inner = idx % pc.INNER;
    const uint token = (idx / pc.INNER) % (pc.T + pc.S);
    const uint outer = idx / (pc.INNER * (pc.T + pc.S));
    if (token < pc.T) {
        output_values[idx] = pe_ctx[(outer * pc.T + token) * pc.INNER + inner];
    } else {
        const uint img_token = token - pc.T;
        output_values[idx] = pe_x[(outer * pc.S + img_token) * pc.INNER + inner];
    }
}
""",
)
