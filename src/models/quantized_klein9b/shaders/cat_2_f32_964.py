"""Generated shader: cat_2_f32_964."""

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


CAT_2_F32_964 = ShaderVariant(
    name='cat_2_f32_964',
    family='export',
    contract=ShaderContract(
        class_name='ExportCat2Program',
        shader_name='cat_2_f32_964',
        fields=(
            TensorFieldSpec(
                name='x0',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('I0_0', 'I0_1', 'I0_2',)),
            ),
            TensorFieldSpec(
                name='x1',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('I1_0', 'I1_1', 'I1_2',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O0', 'O1', 'O2',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=24,
            fields=(
                PushConstantFieldSpec('N_OUT', PushConstantType.UINT32, 0, mul(mul('O0', 'O1'), 'O2'), dynamic=False),
                PushConstantFieldSpec('OUT_STRIDE', PushConstantType.UINT32, 4, mul('O2', 1), dynamic=False),
                PushConstantFieldSpec('END0', PushConstantType.UINT32, 8, mul('I0_2', 1), dynamic=False),
                PushConstantFieldSpec('STRIDE0', PushConstantType.UINT32, 12, mul('I0_2', 1), dynamic=False),
                PushConstantFieldSpec('END1', PushConstantType.UINT32, 16, add(mul('I0_2', 1), mul('I1_2', 1)), dynamic=False),
                PushConstantFieldSpec('STRIDE1', PushConstantType.UINT32, 20, mul('I1_2', 1), dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('O0', 'O1'), 'O2'), 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly X0Buffer { float x0[]; };
layout(set = 0, binding = 1) buffer restrict readonly X1Buffer { float x1[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N_OUT; uint OUT_STRIDE; uint END0; uint STRIDE0; uint END1; uint STRIDE1;  } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N_OUT) {
        uint row = idx / pc.OUT_STRIDE;
        uint col = idx % pc.OUT_STRIDE;
        if (col < pc.END0) {
            output_values[idx] = x0[row * pc.STRIDE0 + (col - 0u)];
        }
        else if (col < pc.END1) {
            output_values[idx] = x1[row * pc.STRIDE1 + (col - pc.END0)];
        }
    }
}
""",
)
