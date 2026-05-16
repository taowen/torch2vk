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
    mul,
)

_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
{{WEIGHT_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { {{WEIGHT_TYPE}} weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { {{BIAS_TYPE}} bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint C; uint H; uint W; uint G; float eps; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float partial_sum[256];
shared float partial_sumsq[256];
void main() {
    const uint group_row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    const uint channels_per_group = pc.C / pc.G;
    const uint cols = channels_per_group * pc.H * pc.W;
    const uint b = group_row / pc.G;
    const uint g = group_row % pc.G;
    if (b >= pc.B) { return; }

    float sum = 0.0;
    float sumsq = 0.0;
    for (uint col = tid; col < cols; col += 256u) {
        const uint hw = col % (pc.H * pc.W);
        const uint channel_in_group = col / (pc.H * pc.W);
        const uint channel = g * channels_per_group + channel_in_group;
        const uint idx = ((b * pc.C + channel) * pc.H * pc.W) + hw;
        const float v = float(x[idx]);
        sum += v;
        sumsq += v * v;
    }
    partial_sum[tid] = sum;
    partial_sumsq[tid] = sumsq;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        barrier();
    }
    const float mean = partial_sum[0] / float(cols);
    const float var = partial_sumsq[0] / float(cols) - mean * mean;
    const float inv_std = inversesqrt(var + pc.eps);

    for (uint col = tid; col < cols; col += 256u) {
        const uint hw = col % (pc.H * pc.W);
        const uint channel_in_group = col / (pc.H * pc.W);
        const uint channel = g * channels_per_group + channel_in_group;
        const uint idx = ((b * pc.C + channel) * pc.H * pc.W) + hw;
        output_values[idx] = {{STORE}};
    }
}
"""


def make_group_norm_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    in_shape = node_input_shape(node, 0)
    out_shape = node_output_shape(node)
    if len(in_shape) != 4 or len(out_shape) != 4:
        return None
    if len(node.args) < 6:
        return None
    num_groups_arg = node.args[1]
    eps_arg = node.args[4]
    if not isinstance(num_groups_arg, int) or not isinstance(eps_arg, float):
        return None
    has_weight = isinstance(node.args[2], Node)
    has_bias = isinstance(node.args[3], Node)
    if not has_weight or not has_bias:
        return None
    in_contract = tuple(f"I{i}" for i in range(4))
    out_contract = tuple(f"O{i}" for i in range(4))
    weight_dtype = node_input_dtype(node, 2)
    bias_dtype = node_input_dtype(node, 3)
    weight_suffix = weight_dtype_suffix(weight_dtype)
    bias_suffix = weight_dtype_suffix(bias_dtype)
    shader_name = f"group_norm_{weight_suffix}w_{bias_suffix}b_f32"
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"ExportGroupNorm{weight_suffix.title()}Weight{bias_suffix.title()}BiasProgram",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=in_contract),
                ),
                TensorFieldSpec(
                    "weight",
                    IOKind.INPUT,
                    "weight",
                    TensorContract(dtype=weight_dtype, shape=("C",)),
                ),
                TensorFieldSpec(
                    "bias",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=bias_dtype, shape=("C",)),
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=out_contract),
                ),
            ),
            push_constants=PushConstantSpec(
                size=24,
                fields=(
                    PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "I0"),
                    PushConstantFieldSpec("C", PushConstantType.UINT32, 4, "I1"),
                    PushConstantFieldSpec("H", PushConstantType.UINT32, 8, "I2"),
                    PushConstantFieldSpec("W", PushConstantType.UINT32, 12, "I3"),
                    PushConstantFieldSpec("G", PushConstantType.UINT32, 16, num_groups_arg),
                    PushConstantFieldSpec("eps", PushConstantType.FLOAT32, 20, eps_arg),
                ),
            ),
            dispatch=(mul("I0", num_groups_arg), 1, 1),
        ),
        source=_source(
            activation_dtype=activation_dtype,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
        ),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _source(*, activation_dtype: str, weight_dtype: str, bias_dtype: str) -> str:
    extension = (
        weight_extension_source("bfloat16") if "bfloat16" in {weight_dtype, bias_dtype} else ""
    )
    return (
        _SOURCE.replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{WEIGHT_EXTENSION}}", extension)
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{WEIGHT_TYPE}}", weight_glsl_type(weight_dtype))
        .replace("{{BIAS_TYPE}}", weight_glsl_type(bias_dtype))
        .replace(
            "{{STORE}}",
            activation_store(
                "(float(x[idx]) - mean) * inv_std * float(weight[channel]) + float(bias[channel])",
                activation_dtype,
            ),
        )
    )
