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
    uint B; uint Ci; uint Li; uint Co; uint Lo; uint Kh;
    int stride; int padding; int dilation;
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
            const int source_pos = int(t) + pc.padding - int(k) * pc.dilation;
            if (source_pos >= 0 && source_pos % pc.stride == 0) {
                const int it_signed = source_pos / pc.stride;
                if (it_signed >= 0 && it_signed < int(pc.Li)) {
                    const uint it = uint(it_signed);
                    const uint x_idx = (b * pc.Ci + ic) * pc.Li + it;
                    const uint w_idx = (ic * pc.Co + oc) * pc.Kh + k;
                    acc = fma(float(x[x_idx]), float(weight[w_idx]), acc);
                }
            }
        }
    }
    output_values[(b * pc.Co + oc) * pc.Lo + t] = {{STORE_ACC}};
}
"""


def make_conv_transpose1d_variant(
    node: Node, activation_dtype: str = "float32"
) -> ShaderVariant | None:
    input_shape = node_input_shape(node, 0)
    weight_shape = node_input_shape(node, 1)
    output_shape = node_output_shape(node)
    if (
        not input_shape
        or not weight_shape
        or not output_shape
        or len(input_shape) != 3
        or len(weight_shape) != 3
        or len(output_shape) != 3
    ):
        return None
    if len(node.args) < 3:
        return None

    stride = _first_int_arg(node, 3, default=1)
    padding = _first_int_arg(node, 4, default=0)
    groups = _first_int_arg(node, 6, default=1)
    dilation = _first_int_arg(node, 7, default=1)
    if groups != 1:
        return None

    weight_dtype = node_input_dtype(node, 1)
    bias_dtype = node_input_dtype(node, 2)
    weight_suffix = weight_dtype_suffix(weight_dtype)
    bias_suffix = weight_dtype_suffix(bias_dtype)
    shader_name = f"conv_transpose1d_{weight_suffix}w_{bias_suffix}b_f32"

    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"ExportConvTranspose1d{weight_suffix.title()}Weight{bias_suffix.title()}BiasProgram",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=("B", "Ci", "Li")),
                ),
                TensorFieldSpec(
                    "weight",
                    IOKind.INPUT,
                    "weight",
                    TensorContract(dtype=weight_dtype, shape=("Ci2", "Co", "Kh")),
                ),
                TensorFieldSpec(
                    "bias", IOKind.INPUT, "input", TensorContract(dtype=bias_dtype, shape=("Co3",))
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=("B", "Co2", "Lo")),
                ),
            ),
            push_constants=PushConstantSpec(
                size=36,
                fields=(
                    PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                    PushConstantFieldSpec("Ci", PushConstantType.UINT32, 4, "Ci"),
                    PushConstantFieldSpec("Li", PushConstantType.UINT32, 8, "Li"),
                    PushConstantFieldSpec("Co", PushConstantType.UINT32, 12, "Co2"),
                    PushConstantFieldSpec("Lo", PushConstantType.UINT32, 16, "Lo"),
                    PushConstantFieldSpec("Kh", PushConstantType.UINT32, 20, "Kh"),
                    PushConstantFieldSpec("stride", PushConstantType.INT32, 24, stride),
                    PushConstantFieldSpec("padding", PushConstantType.INT32, 28, padding),
                    PushConstantFieldSpec("dilation", PushConstantType.INT32, 32, dilation),
                ),
            ),
            dispatch=(ceil_div("Lo", 16), ceil_div("Co2", 16), "B"),
        ),
        source=_source(
            weight_dtype=weight_dtype, bias_dtype=bias_dtype, activation_dtype=activation_dtype
        ),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _first_int_arg(node: Node, index: int, *, default: int) -> int:
    if len(node.args) <= index:
        return default
    arg = node.args[index]
    if isinstance(arg, int):
        return arg
    if isinstance(arg, (list, tuple)) and len(arg) == 1 and isinstance(arg[0], int):
        return arg[0]
    return default


def _source(*, weight_dtype: str, bias_dtype: str, activation_dtype: str) -> str:
    extension = (
        weight_extension_source("bfloat16") if "bfloat16" in {weight_dtype, bias_dtype} else ""
    )
    return (
        _SOURCE_TEMPLATE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{WEIGHT_EXTENSION}}", extension)
        .replace("{{WEIGHT_TYPE}}", weight_glsl_type(weight_dtype))
        .replace("{{BIAS_TYPE}}", weight_glsl_type(bias_dtype))
        .replace("{{STORE_ACC}}", activation_store("acc", activation_dtype))
    )
