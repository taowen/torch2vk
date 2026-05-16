"""Generated shader: conv2d_q8_0w_f32b_f16_2."""

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


CONV2D_Q8_0W_F32B_F16_2 = ShaderVariant(
    name='conv2d_q8_0w_f32b_f16_2',
    family='export',
    contract=ShaderContract(
        class_name='ExportConv2dQ8_0WeightF32BiasProgram',
        shader_name='conv2d_q8_0w_f32b_f16_2',
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
                contract=TensorContract(dtype='uint16', shape=('Co', 2295,), layout=q8_0_halfwords_layout(logical_k=4320, block_size=32, halfwords_per_block=17)),
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
                PushConstantFieldSpec('kh', PushConstantType.UINT32, 28, 3, dynamic=False),
                PushConstantFieldSpec('kw', PushConstantType.UINT32, 32, 3, dynamic=False),
                PushConstantFieldSpec('stride_h', PushConstantType.UINT32, 36, 2, dynamic=False),
                PushConstantFieldSpec('stride_w', PushConstantType.UINT32, 40, 2, dynamic=False),
                PushConstantFieldSpec('pad_h', PushConstantType.UINT32, 44, 1, dynamic=False),
                PushConstantFieldSpec('pad_w', PushConstantType.UINT32, 48, 1, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul('Ho', 'Wo'), 16), ceil_div('Co', 16), 'B'),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
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

void accumulate_q8_0(
    uint b,
    uint oh,
    uint ow,
    uint k,
    float d,
    uint byte_value,
    inout float acc
) {
    const uint ic = k / 9u;
    const uint rem = k - ic * 9u;
    const uint fh = rem / 3u;
    const uint fw = rem - fh * 3u;
    const uint ih_pre = oh * 2u + fh;
    const uint iw_pre = ow * 2u + fw;
    if (ih_pre == 0u || ih_pre > pc.in_h || iw_pre == 0u || iw_pre > pc.in_w) {
        return;
    }
    int quant = int(byte_value);
    if (quant >= 128) { quant -= 256; }
    const uint ih = ih_pre - 1u;
    const uint iw = iw_pre - 1u;
    const uint x_idx = ((b * pc.in_c + ic) * pc.in_h + ih) * pc.in_w + iw;
    acc = fma(float(x[x_idx]), d * float(quant), acc);
}

void main() {
    const uint spatial = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
    const uint oc = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;
    const uint b = gl_WorkGroupID.z;
    if (b >= pc.batch || oc >= pc.out_c || spatial >= pc.out_h * pc.out_w) return;
    const uint oh = spatial / pc.out_w;
    const uint ow = spatial - oh * pc.out_w;
    float acc = float(bias[oc]);
    const uint blocks_per_row = 135u;
    for (uint block = 0u; block < blocks_per_row; ++block) {
        const uint block_half = oc * blocks_per_row * 17u + block * 17u;
        const float d = unpackHalf2x16(uint(weight[block_half])).x;
        const uint base_k = block * 32u;
        for (uint word = 0u; word < 16u; ++word) {
            const uint packed = uint(weight[block_half + 1u + word]);
            const uint k = base_k + word * 2u;
            accumulate_q8_0(b, oh, ow, k, d, packed & 255u, acc);
            accumulate_q8_0(b, oh, ow, k + 1u, d, packed >> 8u, acc);
        }
    }
    output_values[((b * pc.out_c + oc) * pc.out_h + oh) * pc.out_w + ow] = float16_t(acc);
}
""",
)
