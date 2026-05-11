from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    activation_store,
    node_input_dtype,
    node_input_shape,
    node_output_shape,
    weight_dtype_suffix,
    weight_extension_source,
    weight_glsl_type,
)
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

_SOURCE_TEMPLATE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
{{WEIGHT_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { {{WEIGHT_TYPE}} weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { {{BIAS_TYPE}} bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants {
    uint batch; uint in_c; uint in_h; uint in_w;
    uint out_c; uint out_h; uint out_w;
    uint kh; uint kw; uint stride_h; uint stride_w;
    uint pad_h; uint pad_w;
} pc;
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
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
                    const uint w_idx = ((oc * pc.in_c + ic) * pc.kh + fh) * pc.kw + fw;
                    acc = fma(float(x[x_idx]), float(weight[w_idx]), acc);
                }
            }
        }
    }
    output_values[((b * pc.out_c + oc) * pc.out_h + oh) * pc.out_w + ow] = {{STORE_ACC}};
}
"""


def make_conv2d_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
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
            s0, s1 = s
            if isinstance(s0, int) and isinstance(s1, int):
                stride = (s0, s1)
    if len(node.args) >= 5:
        p = node.args[4]
        if isinstance(p, (list, tuple)) and len(p) == 2:
            p0, p1 = p
            if isinstance(p0, int) and isinstance(p1, int):
                padding = (p0, p1)

    has_bias = len(node.args) >= 3 and isinstance(node.args[2], Node)

    x_contract = ("B", "Ci", "Hi", "Wi")
    w_contract = ("Co", "Ci2", "Kh", "Kw")
    out_contract = ("B", "Co2", "Ho", "Wo")

    fields = [
        TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype=activation_dtype, shape=x_contract)),
    ]
    weight_dtype = node_input_dtype(node, 1)
    weight_suffix = weight_dtype_suffix(weight_dtype)
    fields.append(TensorFieldSpec("weight", IOKind.INPUT, "weight", TensorContract(dtype=weight_dtype, shape=w_contract)))
    bias_dtype = weight_dtype
    bias_suffix = weight_suffix
    if has_bias:
        bias_dtype = node_input_dtype(node, 2)
        bias_suffix = weight_dtype_suffix(bias_dtype)
        fields.append(TensorFieldSpec("bias", IOKind.INPUT, "input", TensorContract(dtype=bias_dtype, shape=("Co3",))))
    fields.append(TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype=activation_dtype, shape=out_contract)))
    shader_name = f"conv2d_{weight_suffix}w_{bias_suffix}b_f32"

    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"ExportConv2d{weight_suffix.title()}Weight{bias_suffix.title()}BiasProgram",
            shader_name=shader_name,
            fields=tuple(fields),
            push_constants=PushConstantSpec(
                size=52,
                fields=(
                    PushConstantFieldSpec("batch", PushConstantType.UINT32, 0, "B"),
                    PushConstantFieldSpec("in_c", PushConstantType.UINT32, 4, "Ci"),
                    PushConstantFieldSpec("in_h", PushConstantType.UINT32, 8, "Hi"),
                    PushConstantFieldSpec("in_w", PushConstantType.UINT32, 12, "Wi"),
                    PushConstantFieldSpec("out_c", PushConstantType.UINT32, 16, "Co2"),
                    PushConstantFieldSpec("out_h", PushConstantType.UINT32, 20, "Ho"),
                    PushConstantFieldSpec("out_w", PushConstantType.UINT32, 24, "Wo"),
                    PushConstantFieldSpec("kh", PushConstantType.UINT32, 28, "Kh"),
                    PushConstantFieldSpec("kw", PushConstantType.UINT32, 32, "Kw"),
                    PushConstantFieldSpec("stride_h", PushConstantType.UINT32, 36, stride[0]),
                    PushConstantFieldSpec("stride_w", PushConstantType.UINT32, 40, stride[1]),
                    PushConstantFieldSpec("pad_h", PushConstantType.UINT32, 44, padding[0]),
                    PushConstantFieldSpec("pad_w", PushConstantType.UINT32, 48, padding[1]),
                ),
            ),
            dispatch=(ceil_div(mul("Ho", "Wo"), 16), ceil_div("Co2", 16), "B"),
        ),
        source=_source(weight_dtype=weight_dtype, bias_dtype=bias_dtype, activation_dtype=activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _source(*, weight_dtype: str, bias_dtype: str, activation_dtype: str) -> str:
    extension = (
        weight_extension_source("bfloat16")
        if "bfloat16" in {weight_dtype, bias_dtype}
        else ""
    )
    return (
        _SOURCE_TEMPLATE
        .replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{WEIGHT_EXTENSION}}", extension)
        .replace("{{WEIGHT_TYPE}}", weight_glsl_type(weight_dtype))
        .replace("{{BIAS_TYPE}}", weight_glsl_type(bias_dtype))
        .replace("{{STORE_ACC}}", activation_store("acc", activation_dtype))
    )
