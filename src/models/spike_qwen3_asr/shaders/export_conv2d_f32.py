"""Generated shader: export_conv2d_f32."""

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


EXPORT_CONV2D_F32 = ShaderVariant(
    name='export_conv2d_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportConv2dProgram',
        shader_name='export_conv2d_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('B', 'Ci', 'Hi', 'Wi',)),
            ),
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='float32', shape=('Co', 'Ci2', 'Kh', 'Kw',)),
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
                contract=TensorContract(dtype='float32', shape=('B', 'Co2', 'Ho', 'Wo',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=52,
            fields=(
                PushConstantFieldSpec('batch', PushConstantType.UINT32, 0, 11, dynamic=False),
                PushConstantFieldSpec('in_c', PushConstantType.UINT32, 4, 1, dynamic=False),
                PushConstantFieldSpec('in_h', PushConstantType.UINT32, 8, 128, dynamic=False),
                PushConstantFieldSpec('in_w', PushConstantType.UINT32, 12, 100, dynamic=False),
                PushConstantFieldSpec('out_c', PushConstantType.UINT32, 16, 480, dynamic=False),
                PushConstantFieldSpec('out_h', PushConstantType.UINT32, 20, 64, dynamic=False),
                PushConstantFieldSpec('out_w', PushConstantType.UINT32, 24, 50, dynamic=False),
                PushConstantFieldSpec('kh', PushConstantType.UINT32, 28, 3, dynamic=False),
                PushConstantFieldSpec('kw', PushConstantType.UINT32, 32, 3, dynamic=False),
                PushConstantFieldSpec('stride_h', PushConstantType.UINT32, 36, 2, dynamic=False),
                PushConstantFieldSpec('stride_w', PushConstantType.UINT32, 40, 2, dynamic=False),
                PushConstantFieldSpec('pad_h', PushConstantType.UINT32, 44, 1, dynamic=False),
                PushConstantFieldSpec('pad_w', PushConstantType.UINT32, 48, 1, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(16896000, 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { float bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants {
    uint batch; uint in_c; uint in_h; uint in_w;
    uint out_c; uint out_h; uint out_w;
    uint kh; uint kw; uint stride_h; uint stride_w;
    uint pad_h; uint pad_w;
} pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.batch * pc.out_c * pc.out_h * pc.out_w;
    if (idx >= total) return;
    uint rem = idx;
    const uint ow = rem % pc.out_w; rem /= pc.out_w;
    const uint oh = rem % pc.out_h; rem /= pc.out_h;
    const uint oc = rem % pc.out_c; rem /= pc.out_c;
    const uint b = rem;
    float acc = bias[oc];
    for (uint ic = 0u; ic < pc.in_c; ++ic) {
        for (uint fh = 0u; fh < pc.kh; ++fh) {
            for (uint fw = 0u; fw < pc.kw; ++fw) {
                const uint ih = oh * pc.stride_h + fh - pc.pad_h;
                const uint iw = ow * pc.stride_w + fw - pc.pad_w;
                if (ih < pc.in_h && iw < pc.in_w) {
                    const uint x_idx = ((b * pc.in_c + ic) * pc.in_h + ih) * pc.in_w + iw;
                    const uint w_idx = ((oc * pc.in_c + ic) * pc.kh + fh) * pc.kw + fw;
                    acc += x[x_idx] * weight[w_idx];
                }
            }
        }
    }
    output_values[idx] = acc;
}
""",
)
