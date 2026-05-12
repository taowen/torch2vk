"""Generated shader: conv_transpose1d_q8_0w_f32b_f32."""

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
from torch2vk.vulkan.types import (
    q8_0_halfwords_layout,
)


CONV_TRANSPOSE1D_Q8_0W_F32B_F32 = ShaderVariant(
    name='conv_transpose1d_q8_0w_f32b_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportConvTranspose1dQ8_0WeightF32BiasProgram',
        shader_name='conv_transpose1d_q8_0w_f32b_f32',
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
                contract=TensorContract(dtype='uint16', shape=('Ci', 4352,), layout=q8_0_halfwords_layout(logical_k=8192, block_size=32, halfwords_per_block=17)),
            ),
            TensorFieldSpec(
                name='bias',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('Co',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('B', 'Co', 'Lo',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=36,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('Ci', PushConstantType.UINT32, 4, 'Ci', dynamic=False),
                PushConstantFieldSpec('Li', PushConstantType.UINT32, 8, 'Li', dynamic=False),
                PushConstantFieldSpec('Co', PushConstantType.UINT32, 12, 'Co', dynamic=False),
                PushConstantFieldSpec('Lo', PushConstantType.UINT32, 16, 'Lo', dynamic=False),
                PushConstantFieldSpec('Kh', PushConstantType.UINT32, 20, 16, dynamic=False),
                PushConstantFieldSpec('stride', PushConstantType.INT32, 24, 8, dynamic=False),
                PushConstantFieldSpec('padding', PushConstantType.INT32, 28, 4, dynamic=False),
                PushConstantFieldSpec('dilation', PushConstantType.INT32, 32, 1, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div('Lo', 16), ceil_div('Co', 16), 'B'),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { float bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants {
    uint B; uint Ci; uint Li; uint Co; uint Lo; uint Kh;
    int stride; int padding; int dilation;
} pc;
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

float q8_0_value(uint row, uint k) {
    const uint kernel_k = pc.Co * pc.Kh;
    const uint blocks_per_row = (kernel_k + 31u) / 32u;
    const uint block_index = k >> 5u;
    const uint block_half = row * blocks_per_row * 17u + block_index * 17u;
    const float d = unpackHalf2x16(uint(weight[block_half])).x;
    const uint local = k & 31u;
    const uint packed = uint(weight[block_half + 1u + (local >> 1u)]);
    uint byte_value = ((local & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
    int quant = int(byte_value);
    if (quant >= 128) { quant -= 256; }
    return d * float(quant);
}

void main() {
    const uint t = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
    const uint oc = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;
    const uint b = gl_WorkGroupID.z;
    if (b >= pc.B || oc >= pc.Co || t >= pc.Lo) return;
    float acc = float(bias[oc]);
    for (uint ic = 0u; ic < pc.Ci; ++ic) {
        for (uint k = 0u; k < pc.Kh; ++k) {
            const int source_pos = int(t) + pc.padding - int(k) * pc.dilation;
            if (source_pos >= 0 && source_pos % pc.stride == 0) {
                const int it_signed = source_pos / pc.stride;
                if (it_signed >= 0 && it_signed < int(pc.Li)) {
                    const uint it = uint(it_signed);
                    const uint x_idx = (b * pc.Ci + ic) * pc.Li + it;
                    const uint weight_k = oc * pc.Kh + k;
                    acc = fma(float(x[x_idx]), q8_0_value(ic, weight_k), acc);
                }
            }
        }
    }
    output_values[(b * pc.Co + oc) * pc.Lo + t] = float16_t(acc);
}
""",
)
