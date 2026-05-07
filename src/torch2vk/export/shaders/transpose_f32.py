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
    ceil_div,
)

_SOURCE_TRANSPOSE_12_4D = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint D1; uint D2; uint D3; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        // output layout: (B, D2, D1, D3) — transposed dims 1 and 2
        uint d3 = idx % pc.D3;
        uint rem = idx / pc.D3;
        uint d1 = rem % pc.D1;
        rem = rem / pc.D1;
        uint d2 = rem % pc.D2;
        uint b = rem / pc.D2;
        // input layout: (B, D1, D2, D3)
        uint in_idx = ((b * pc.D1 + d1) * pc.D2 + d2) * pc.D3 + d3;
        output_values[idx] = x[in_idx];
    }
}
"""


def make_transpose_variant(node: Node) -> ShaderVariant | None:
    in_shape = node_input_shape(node, 0)
    out_shape = node_output_shape(node)
    if not in_shape or not out_shape:
        return None

    dim0 = int(node.args[1]) if len(node.args) > 1 and isinstance(node.args[1], int) else 0
    dim1 = int(node.args[2]) if len(node.args) > 2 and isinstance(node.args[2], int) else 1

    if dim0 > dim1:
        dim0, dim1 = dim1, dim0

    n_total = math.prod(out_shape)

    if len(in_shape) == 4 and dim0 == 1 and dim1 == 2:
        d1 = in_shape[1]
        d2 = in_shape[2]
        d3 = in_shape[3]
    else:
        d1 = in_shape[dim0]
        d2 = in_shape[dim1]
        d3 = 1
        for d in in_shape[dim1 + 1:]:
            d3 *= d

    in_contract = tuple(f"I{i}" for i in range(len(in_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))

    return ShaderVariant(
        name="export_transpose_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportTransposeProgram",
            shader_name="export_transpose_f32",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=in_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=16,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_total),
                    PushConstantFieldSpec("D1", PushConstantType.UINT32, 4, d1),
                    PushConstantFieldSpec("D2", PushConstantType.UINT32, 8, d2),
                    PushConstantFieldSpec("D3", PushConstantType.UINT32, 12, d3),
                ),
            ),
            dispatch=(ceil_div(n_total, 256), 1, 1),
        ),
        source=_SOURCE_TRANSPOSE_12_4D,
    )
