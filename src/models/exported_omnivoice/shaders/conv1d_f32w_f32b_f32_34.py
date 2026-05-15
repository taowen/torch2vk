"""Generated shader: conv1d_f32w_f32b_f32_34."""

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


CONV1D_F32W_F32B_F32_34 = ShaderVariant(
    name='conv1d_f32w_f32b_f32_34',
    family='export',
    contract=ShaderContract(
        class_name='ExportConv1dF32WeightF32BiasProgram',
        shader_name='conv1d_f32w_f32b_f32_34',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('B', 'Ci', 'Li',)),
            ),
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='float32', shape=('Co', 'Ci2', 'Kh',)),
            ),
            TensorFieldSpec(
                name='bias',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('Co3',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('B', 'Co2', 'Lo',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=36,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('Ci', PushConstantType.UINT32, 4, 'Ci', dynamic=False),
                PushConstantFieldSpec('Li', PushConstantType.UINT32, 8, 'Li', dynamic=False),
                PushConstantFieldSpec('Co', PushConstantType.UINT32, 12, 'Co2', dynamic=False),
                PushConstantFieldSpec('Lo', PushConstantType.UINT32, 16, 'Lo', dynamic=False),
                PushConstantFieldSpec('Kh', PushConstantType.UINT32, 20, 'Kh', dynamic=False),
                PushConstantFieldSpec('stride', PushConstantType.UINT32, 24, 1, dynamic=False),
                PushConstantFieldSpec('padding', PushConstantType.UINT32, 28, 9, dynamic=False),
                PushConstantFieldSpec('dilation', PushConstantType.UINT32, 32, 3, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div('Lo', 16), ceil_div('Co2', 16), 'B'),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { float bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants {
    uint B; uint Ci; uint Li; uint Co; uint Lo; uint Kh;
    uint stride; uint padding; uint dilation;
} pc;
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main() {
    const uint t = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
    const uint oc = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;
    const uint b = gl_WorkGroupID.z;
    if (b >= pc.B || oc >= pc.Co || t >= pc.Lo) return;
    float acc = float(bias[oc]);
    for (uint ic = 0u; ic < pc.Ci; ++ic) {
        for (uint k = 0u; k < pc.Kh; ++k) {
            const uint padded_pos = t * pc.stride + k * pc.dilation;
            if (padded_pos >= pc.padding) {
                const uint it = padded_pos - pc.padding;
                if (it < pc.Li) {
                    const uint x_idx = (b * pc.Ci + ic) * pc.Li + it;
                    const uint w_idx = (oc * pc.Ci + ic) * pc.Kh + k;
                    acc = fma(float(x[x_idx]), float(weight[w_idx]), acc);
                }
            }
        }
    }
    output_values[(b * pc.Co + oc) * pc.Lo + t] = float16_t(acc);
}
""",
)
