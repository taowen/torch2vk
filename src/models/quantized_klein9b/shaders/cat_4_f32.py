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
    add,
    ceil_div,
    mul,
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
                contract=TensorContract(dtype='float32', shape=('I0_0', 'I0_1', 'I0_2', 'I0_3', 'I0_4',)),
            ),
            TensorFieldSpec(
                name='x1',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('I1_0', 'I1_1', 'I1_2', 'I1_3', 'I1_4',)),
            ),
            TensorFieldSpec(
                name='x2',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('I2_0', 'I2_1', 'I2_2', 'I2_3', 'I2_4',)),
            ),
            TensorFieldSpec(
                name='x3',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('I3_0', 'I3_1', 'I3_2', 'I3_3', 'I3_4',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O0', 'O1', 'O2', 'O3', 'O4',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=40,
            fields=(
                PushConstantFieldSpec('N_OUT', PushConstantType.UINT32, 0, mul(mul(mul(mul('O0', 'O1'), 'O2'), 'O3'), 'O4'), dynamic=False),
                PushConstantFieldSpec('OUT_STRIDE', PushConstantType.UINT32, 4, mul('O2', mul('O3', 'O4')), dynamic=False),
                PushConstantFieldSpec('END0', PushConstantType.UINT32, 8, mul('I0_2', mul('I0_3', 'I0_4')), dynamic=False),
                PushConstantFieldSpec('STRIDE0', PushConstantType.UINT32, 12, mul('I0_2', mul('I0_3', 'I0_4')), dynamic=False),
                PushConstantFieldSpec('END1', PushConstantType.UINT32, 16, add(mul('I0_2', mul('I0_3', 'I0_4')), mul('I1_2', mul('I1_3', 'I1_4'))), dynamic=False),
                PushConstantFieldSpec('STRIDE1', PushConstantType.UINT32, 20, mul('I1_2', mul('I1_3', 'I1_4')), dynamic=False),
                PushConstantFieldSpec('END2', PushConstantType.UINT32, 24, add(add(mul('I0_2', mul('I0_3', 'I0_4')), mul('I1_2', mul('I1_3', 'I1_4'))), mul('I2_2', mul('I2_3', 'I2_4'))), dynamic=False),
                PushConstantFieldSpec('STRIDE2', PushConstantType.UINT32, 28, mul('I2_2', mul('I2_3', 'I2_4')), dynamic=False),
                PushConstantFieldSpec('END3', PushConstantType.UINT32, 32, add(add(add(mul('I0_2', mul('I0_3', 'I0_4')), mul('I1_2', mul('I1_3', 'I1_4'))), mul('I2_2', mul('I2_3', 'I2_4'))), mul('I3_2', mul('I3_3', 'I3_4'))), dynamic=False),
                PushConstantFieldSpec('STRIDE3', PushConstantType.UINT32, 36, mul('I3_2', mul('I3_3', 'I3_4')), dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul(mul(mul('O0', 'O1'), 'O2'), 'O3'), 'O4'), 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly X0Buffer { float x0[]; };
layout(set = 0, binding = 1) buffer restrict readonly X1Buffer { float x1[]; };
layout(set = 0, binding = 2) buffer restrict readonly X2Buffer { float x2[]; };
layout(set = 0, binding = 3) buffer restrict readonly X3Buffer { float x3[]; };
layout(set = 0, binding = 4) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N_OUT; uint OUT_STRIDE; uint END0; uint STRIDE0; uint END1; uint STRIDE1; uint END2; uint STRIDE2; uint END3; uint STRIDE3;  } pc;
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
        else if (col < pc.END2) {
            output_values[idx] = x2[row * pc.STRIDE2 + (col - pc.END1)];
        }
        else if (col < pc.END3) {
            output_values[idx] = x3[row * pc.STRIDE3 + (col - pc.END2)];
        }
    }
}
""",
)
