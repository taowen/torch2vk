from __future__ import annotations

import hashlib

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
    mul,
)

_SOURCE = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndexBuffer { int index_values[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint O; uint H; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.O * pc.H;
    if (idx >= total) { return; }

    const uint row = idx / pc.H;
    const uint h = idx - row * pc.H;
    const uint src_row = uint(index_values[row]);
    output_values[idx] = x[src_row * pc.H + h];
}
"""


def make_index_select_variant(node: Node) -> ShaderVariant | None:
    if len(node.args) != 3:
        return None
    dim = node.args[1]
    if not isinstance(dim, int) or dim != 0:
        return None
    in_shape = node_input_shape(node, 0)
    index_shape = node_input_shape(node, 2)
    out_shape = node_output_shape(node)
    if len(in_shape) != 2 or len(index_shape) != 1 or len(out_shape) != 2:
        return None
    if out_shape[0] != index_shape[0] or out_shape[1] != in_shape[1]:
        return None

    shader_name = _shader_name(in_shape, index_shape, out_shape, dim)
    total = mul("O", "H")
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name="ExportIndexSelectF32Program",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("N", "H"))),
                TensorFieldSpec("index", IOKind.INPUT, "index", TensorContract(dtype="int32", shape=("O",))),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=("O", "H"))),
            ),
            push_constants=PushConstantSpec(
                size=8,
                fields=(
                    PushConstantFieldSpec("O", PushConstantType.UINT32, 0, "O"),
                    PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
                ),
            ),
            dispatch=(ceil_div(total, 256), 1, 1),
        ),
        source=_SOURCE,
    )


def _shader_name(
    in_shape: tuple[int, ...],
    index_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dim: int,
) -> str:
    digest = hashlib.sha1(repr((in_shape, index_shape, out_shape, dim)).encode()).hexdigest()[:10]
    return f"export_index_select_f32_{digest}"
