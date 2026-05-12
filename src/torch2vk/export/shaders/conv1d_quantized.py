from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_store,
    activation_variant_name,
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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements
from torch2vk.vulkan.types import q8_0_halfwords_layout

_SOURCE_TEMPLATE = """\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
{{ACTIVATION_EXTENSION}}\
{{BIAS_EXTENSION}}\

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { {{BIAS_TYPE}} bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants {
    uint B; uint Ci; uint Li; uint Co; uint Lo; uint Kh;
    uint stride; uint padding; uint dilation;
} pc;
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

float q8_0_value(uint row, uint k) {
    const uint kernel_k = pc.Ci * pc.Kh;
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
            const uint padded_pos = t * pc.stride + k * pc.dilation;
            if (padded_pos >= pc.padding) {
                const uint it = padded_pos - pc.padding;
                if (it < pc.Li) {
                    const uint x_idx = (b * pc.Ci + ic) * pc.Li + it;
                    const uint weight_k = ic * pc.Kh + k;
                    acc = fma(float(x[x_idx]), q8_0_value(oc, weight_k), acc);
                }
            }
        }
    }
    output_values[(b * pc.Co + oc) * pc.Lo + t] = {{STORE_ACC}};
}
"""


def make_conv1d_q8_0_variant(node: Node, activation_dtype: str = "float16") -> ShaderVariant | None:
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
    dilation = _first_int_arg(node, 5, default=1)
    groups = _first_int_arg(node, 6, default=1)
    if groups != 1:
        return None

    _, in_channels, kernel_width = weight_shape
    kernel_width = int(kernel_width)
    kernel_k = int(in_channels) * kernel_width
    blocks_halfwords = ((kernel_k + 31) // 32) * 17

    bias_dtype = node_input_dtype(node, 2)
    bias_suffix = weight_dtype_suffix(bias_dtype)
    base_name = f"conv1d_q8_0w_{bias_suffix}b_f32"
    shader_name = activation_variant_name(base_name, activation_dtype)

    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"ExportConv1dQ8_0Weight{bias_suffix.title()}BiasProgram",
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
                    TensorContract(
                        dtype="uint16",
                        shape=("Co", blocks_halfwords),
                        layout=q8_0_halfwords_layout(logical_k=kernel_k),
                    ),
                ),
                TensorFieldSpec(
                    "bias", IOKind.INPUT, "input", TensorContract(dtype=bias_dtype, shape=("Co",))
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=("B", "Co", "Lo")),
                ),
            ),
            push_constants=PushConstantSpec(
                size=36,
                fields=(
                    PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                    PushConstantFieldSpec("Ci", PushConstantType.UINT32, 4, "Ci"),
                    PushConstantFieldSpec("Li", PushConstantType.UINT32, 8, "Li"),
                    PushConstantFieldSpec("Co", PushConstantType.UINT32, 12, "Co"),
                    PushConstantFieldSpec("Lo", PushConstantType.UINT32, 16, "Lo"),
                    PushConstantFieldSpec("Kh", PushConstantType.UINT32, 20, kernel_width),
                    PushConstantFieldSpec("stride", PushConstantType.UINT32, 24, stride),
                    PushConstantFieldSpec("padding", PushConstantType.UINT32, 28, padding),
                    PushConstantFieldSpec("dilation", PushConstantType.UINT32, 32, dilation),
                ),
            ),
            dispatch=(ceil_div("Lo", 16), ceil_div("Co", 16), "B"),
        ),
        source=_source(bias_dtype=bias_dtype, activation_dtype=activation_dtype),
        execution_requirements=ShaderExecutionRequirements(
            require_storage_buffer_16bit_access=True
        ),
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


def _source(*, bias_dtype: str, activation_dtype: str) -> str:
    return (
        _SOURCE_TEMPLATE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace(
            "{{BIAS_EXTENSION}}",
            weight_extension_source("bfloat16") if bias_dtype == "bfloat16" else "",
        )
        .replace("{{BIAS_TYPE}}", weight_glsl_type(bias_dtype))
        .replace("{{STORE_ACC}}", activation_store("acc", activation_dtype))
    )
