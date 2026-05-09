from __future__ import annotations

import math

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
)

_SOURCE = """\
#version 450
#extension GL_EXT_bfloat16 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { bfloat16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer { bfloat16_t bias[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint ROWS; uint COLS; float eps; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float partial_sum[256];
shared float partial_sumsq[256];
void main() {
    const uint row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    if (row >= pc.ROWS) { return; }
    float sum = 0.0;
    float sumsq = 0.0;
    for (uint c = tid; c < pc.COLS; c += 256u) {
        float v = x[row * pc.COLS + c];
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
    float mean = partial_sum[0] / float(pc.COLS);
    float var = partial_sumsq[0] / float(pc.COLS) - mean * mean;
    float inv_std = inversesqrt(var + pc.eps);
    for (uint c = tid; c < pc.COLS; c += 256u) {
        uint idx = row * pc.COLS + c;
        output_values[idx] = fma((x[idx] - mean) * inv_std, weight[c], bias[c]);
    }
}
"""


def make_layer_norm_variant(node: Node) -> ShaderVariant | None:
    in_shape = node_input_shape(node, 0)
    out_shape = node_output_shape(node)
    if not in_shape or not out_shape:
        return None

    normalized_shape_arg = node.args[1] if len(node.args) > 1 else None
    if isinstance(normalized_shape_arg, (list, tuple)):
        normalized_dims = [d for d in normalized_shape_arg if isinstance(d, int)]
        if len(normalized_dims) == len(normalized_shape_arg):
            cols = math.prod(normalized_dims)
        else:
            cols = in_shape[-1]
    elif isinstance(normalized_shape_arg, int):
        cols = normalized_shape_arg
    else:
        cols = in_shape[-1]

    rows = math.prod(in_shape) // cols

    eps_val = 1e-5
    if len(node.args) > 4 and isinstance(node.args[4], (int, float)):
        eps_val = float(node.args[4])
    else:
        eps_arg = node.kwargs.get("eps")
        if isinstance(eps_arg, (int, float)):
            eps_val = float(eps_arg)

    in_contract = tuple(f"I{i}" for i in range(len(in_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))

    return ShaderVariant(
        name="layer_norm_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportLayerNormProgram",
            shader_name="layer_norm_f32",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=in_contract)),
                TensorFieldSpec("weight", IOKind.INPUT, "weight", TensorContract(dtype="bfloat16", shape=("W0",))),
                TensorFieldSpec("bias", IOKind.INPUT, "input", TensorContract(dtype="bfloat16", shape=("W0",))),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=12,
                fields=(
                    PushConstantFieldSpec("ROWS", PushConstantType.UINT32, 0, rows),
                    PushConstantFieldSpec("COLS", PushConstantType.UINT32, 4, cols),
                    PushConstantFieldSpec("eps", PushConstantType.FLOAT32, 8, eps_val),
                ),
            ),
            dispatch=(rows, 1, 1),
        ),
        source=_SOURCE,
    )
