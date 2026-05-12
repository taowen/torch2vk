from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
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
{{BIAS_BUFFER}}layout(set = 0, binding = {{OUTPUT_BINDING}}) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
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
    float acc = {{ACC_INIT}};
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
    output_values[((b * pc.out_c + oc) * pc.out_h + oh) * pc.out_w + ow] = {{STORE_ACC}};
}
"""


def make_conv2d_q8_0_variant(node: Node, activation_dtype: str = "float16") -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    w_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not x_shape or not w_shape or not out_shape or len(x_shape) != 4 or len(w_shape) != 4:
        return None

    _, in_c, _, _ = x_shape
    out_c, _, kh, kw = w_shape
    kernel_k = int(in_c) * int(kh) * int(kw)
    blocks_halfwords = ((kernel_k + 31) // 32) * 17

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
    fields: list[TensorFieldSpec] = [
        TensorFieldSpec(
            "x",
            IOKind.INPUT,
            "input",
            TensorContract(dtype=activation_dtype, shape=("B", "Ci", "Hi", "Wi")),
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
    ]
    bias_dtype = "float32"
    bias_suffix = "nobias"
    if has_bias:
        bias_dtype = node_input_dtype(node, 2)
        bias_suffix = weight_dtype_suffix(bias_dtype)
        fields.append(
            TensorFieldSpec(
                "bias", IOKind.INPUT, "input", TensorContract(dtype=bias_dtype, shape=("Co",))
            )
        )
    fields.append(
        TensorFieldSpec(
            "output",
            IOKind.OUTPUT,
            "output",
            TensorContract(dtype=activation_dtype, shape=("B", "Co", "Ho", "Wo")),
        )
    )

    shader_name = f"conv2d_q8_0w_{bias_suffix}b_{'f16' if activation_dtype == 'float16' else 'f32'}"
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"ExportConv2dQ8_0Weight{bias_suffix.title()}BiasProgram",
            shader_name=shader_name,
            fields=tuple(fields),
            push_constants=PushConstantSpec(
                size=52,
                fields=(
                    PushConstantFieldSpec("batch", PushConstantType.UINT32, 0, "B"),
                    PushConstantFieldSpec("in_c", PushConstantType.UINT32, 4, "Ci"),
                    PushConstantFieldSpec("in_h", PushConstantType.UINT32, 8, "Hi"),
                    PushConstantFieldSpec("in_w", PushConstantType.UINT32, 12, "Wi"),
                    PushConstantFieldSpec("out_c", PushConstantType.UINT32, 16, "Co"),
                    PushConstantFieldSpec("out_h", PushConstantType.UINT32, 20, "Ho"),
                    PushConstantFieldSpec("out_w", PushConstantType.UINT32, 24, "Wo"),
                    PushConstantFieldSpec("kh", PushConstantType.UINT32, 28, kh),
                    PushConstantFieldSpec("kw", PushConstantType.UINT32, 32, kw),
                    PushConstantFieldSpec("stride_h", PushConstantType.UINT32, 36, stride[0]),
                    PushConstantFieldSpec("stride_w", PushConstantType.UINT32, 40, stride[1]),
                    PushConstantFieldSpec("pad_h", PushConstantType.UINT32, 44, padding[0]),
                    PushConstantFieldSpec("pad_w", PushConstantType.UINT32, 48, padding[1]),
                ),
            ),
            dispatch=(ceil_div(mul("Ho", "Wo"), 16), ceil_div("Co", 16), "B"),
        ),
        execution_requirements=ShaderExecutionRequirements(
            require_storage_buffer_16bit_access=True
        ),
        source=_source(has_bias=has_bias, bias_dtype=bias_dtype, activation_dtype=activation_dtype),
    )


def _source(*, has_bias: bool, bias_dtype: str, activation_dtype: str) -> str:
    if has_bias:
        bias_buffer = f"layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer {{ {weight_glsl_type(bias_dtype)} bias[]; }};\n"
        output_binding = "3"
        acc_init = "float(bias[oc])"
    else:
        bias_buffer = ""
        output_binding = "2"
        acc_init = "0.0"
    return (
        _SOURCE_TEMPLATE.replace(
            "{{ACTIVATION_EXTENSION}}",
            activation_extension_source(activation_dtype),
        )
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_ACC}}", activation_store("acc", activation_dtype))
        .replace(
            "{{BIAS_EXTENSION}}",
            weight_extension_source("bfloat16") if bias_dtype == "bfloat16" else "",
        )
        .replace("{{BIAS_BUFFER}}", bias_buffer)
        .replace("{{OUTPUT_BINDING}}", output_binding)
        .replace("{{ACC_INIT}}", acc_init)
    )
