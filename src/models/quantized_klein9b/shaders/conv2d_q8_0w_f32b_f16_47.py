"""Generated shader: conv2d_q8_0w_f32b_f16_47."""

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
from torch2vk.vulkan.types import (
    q8_0_halfwords_layout,
)


CONV2D_Q8_0W_F32B_F16_47 = ShaderVariant(
    name='conv2d_q8_0w_f32b_f16_47',
    family='export',
    contract=ShaderContract(
        class_name='ExportConv2dQ8_0WeightF32BiasProgram',
        shader_name='conv2d_q8_0w_f32b_f16_47',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('B', 'Ci', 'Hi', 'Wi',)),
            ),
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='uint16', shape=('Co', 136,), layout=q8_0_halfwords_layout(logical_k=256, block_size=32, halfwords_per_block=17)),
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
                contract=TensorContract(dtype='float16', shape=('B', 'Co', 'Ho', 'Wo',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=52,
            fields=(
                PushConstantFieldSpec('batch', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('in_c', PushConstantType.UINT32, 4, 'Ci', dynamic=False),
                PushConstantFieldSpec('in_h', PushConstantType.UINT32, 8, 'Hi', dynamic=False),
                PushConstantFieldSpec('in_w', PushConstantType.UINT32, 12, 'Wi', dynamic=False),
                PushConstantFieldSpec('out_c', PushConstantType.UINT32, 16, 'Co', dynamic=False),
                PushConstantFieldSpec('out_h', PushConstantType.UINT32, 20, 'Ho', dynamic=False),
                PushConstantFieldSpec('out_w', PushConstantType.UINT32, 24, 'Wo', dynamic=False),
                PushConstantFieldSpec('kh', PushConstantType.UINT32, 28, 1, dynamic=False),
                PushConstantFieldSpec('kw', PushConstantType.UINT32, 32, 1, dynamic=False),
                PushConstantFieldSpec('stride_h', PushConstantType.UINT32, 36, 1, dynamic=False),
                PushConstantFieldSpec('stride_w', PushConstantType.UINT32, 40, 1, dynamic=False),
                PushConstantFieldSpec('pad_h', PushConstantType.UINT32, 44, 0, dynamic=False),
                PushConstantFieldSpec('pad_w', PushConstantType.UINT32, 48, 0, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul('Ho', 'Wo'), 16), ceil_div('Co', 16), 'B'),
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
    uint batch; uint in_c; uint in_h; uint in_w;
    uint out_c; uint out_h; uint out_w;
    uint kh; uint kw; uint stride_h; uint stride_w;
    uint pad_h; uint pad_w;
} pc;
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

float q8_0_value(uint row, uint k) {
    const uint kernel_k = pc.in_c * pc.kh * pc.kw;
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
    const uint spatial = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
    const uint oc = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;
    const uint b = gl_WorkGroupID.z;
    if (b >= pc.batch || oc >= pc.out_c || spatial >= pc.out_h * pc.out_w) return;
    const uint oh = spatial / pc.out_w;
    const uint ow = spatial - oh * pc.out_w;
    float acc = float(bias[oc]);
    for (uint ic = 0u; ic < pc.in_c; ++ic) {
        for (uint fh = 0u; fh < pc.kh; ++fh) {
            for (uint fw = 0u; fw < pc.kw; ++fw) {
                const uint ih = oh * pc.stride_h + fh - pc.pad_h;
                const uint iw = ow * pc.stride_w + fw - pc.pad_w;
                if (ih < pc.in_h && iw < pc.in_w) {
                    const uint x_idx = ((b * pc.in_c + ic) * pc.in_h + ih) * pc.in_w + iw;
                    const uint k = ((ic * pc.kh + fh) * pc.kw + fw);
                    acc = fma(float(x[x_idx]), q8_0_value(oc, k), acc);
                }
            }
        }
    }
    output_values[((b * pc.out_c + oc) * pc.out_h + oh) * pc.out_w + ow] = float16_t(acc);
}
""",
)
