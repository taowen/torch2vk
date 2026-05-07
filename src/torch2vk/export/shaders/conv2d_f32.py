from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import node_input_shape, node_output_shape
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

_SOURCE = """\
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
"""


def make_conv2d_variant(node: Node) -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    w_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not x_shape or not w_shape or not out_shape or len(x_shape) != 4:
        return None

    batch, in_c, in_h, in_w = x_shape
    out_c, _, kh, kw = w_shape
    _, _, out_h, out_w = out_shape

    stride = (1, 1)
    padding = (0, 0)
    if len(node.args) >= 4:
        s = node.args[3]
        if isinstance(s, (list, tuple)) and len(s) == 2:
            stride = (int(s[0]), int(s[1]))
    if len(node.args) >= 5:
        p = node.args[4]
        if isinstance(p, (list, tuple)) and len(p) == 2:
            padding = (int(p[0]), int(p[1]))

    has_bias = len(node.args) >= 3 and isinstance(node.args[2], Node)

    x_contract = ("B", "Ci", "Hi", "Wi")
    w_contract = ("Co", "Ci2", "Kh", "Kw")
    out_contract = ("B", "Co2", "Ho", "Wo")

    fields = [
        TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=x_contract)),
        TensorFieldSpec("weight", IOKind.INPUT, "weight", TensorContract(dtype="float32", shape=w_contract)),
    ]
    if has_bias:
        fields.append(TensorFieldSpec("bias", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("Co3",))))
    fields.append(TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)))

    total = batch * out_c * out_h * out_w
    return ShaderVariant(
        name="export_conv2d_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportConv2dProgram",
            shader_name="export_conv2d_f32",
            fields=tuple(fields),
            push_constants=PushConstantSpec(
                size=52,
                fields=(
                    PushConstantFieldSpec("batch", PushConstantType.UINT32, 0, batch),
                    PushConstantFieldSpec("in_c", PushConstantType.UINT32, 4, in_c),
                    PushConstantFieldSpec("in_h", PushConstantType.UINT32, 8, in_h),
                    PushConstantFieldSpec("in_w", PushConstantType.UINT32, 12, in_w),
                    PushConstantFieldSpec("out_c", PushConstantType.UINT32, 16, out_c),
                    PushConstantFieldSpec("out_h", PushConstantType.UINT32, 20, out_h),
                    PushConstantFieldSpec("out_w", PushConstantType.UINT32, 24, out_w),
                    PushConstantFieldSpec("kh", PushConstantType.UINT32, 28, kh),
                    PushConstantFieldSpec("kw", PushConstantType.UINT32, 32, kw),
                    PushConstantFieldSpec("stride_h", PushConstantType.UINT32, 36, stride[0]),
                    PushConstantFieldSpec("stride_w", PushConstantType.UINT32, 40, stride[1]),
                    PushConstantFieldSpec("pad_h", PushConstantType.UINT32, 44, padding[0]),
                    PushConstantFieldSpec("pad_w", PushConstantType.UINT32, 48, padding[1]),
                ),
            ),
            dispatch=(ceil_div(total, 256), 1, 1),
        ),
        source=_SOURCE,
    )
