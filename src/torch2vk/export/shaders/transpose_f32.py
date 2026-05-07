from __future__ import annotations

import hashlib
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

_SOURCE_TEMPLATE = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        uint rem = idx;
__DECODE_OUTPUT_COORDS__
        uint in_idx = 0u;
__ENCODE_INPUT_INDEX__
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

    rank = len(in_shape)
    if dim0 < 0:
        dim0 += rank
    if dim1 < 0:
        dim1 += rank
    if not (0 <= dim0 < rank and 0 <= dim1 < rank):
        return None
    if dim0 > dim1:
        dim0, dim1 = dim1, dim0

    n_total = math.prod(out_shape)

    in_contract = tuple(f"I{i}" for i in range(len(in_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))
    shader_name = _shader_name(in_shape, out_shape, dim0, dim1)

    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name="ExportTransposeProgram",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=in_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=4,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_total),
                ),
            ),
            dispatch=(ceil_div(n_total, 256), 1, 1),
        ),
        source=_transpose_source(in_shape, out_shape, dim0, dim1),
    )


def _shader_name(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dim0: int,
    dim1: int,
) -> str:
    payload = f"{in_shape}->{out_shape}:{dim0}:{dim1}".encode()
    digest = hashlib.sha1(payload).hexdigest()[:10]
    return f"export_transpose_f32_{digest}"


def _transpose_source(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dim0: int,
    dim1: int,
) -> str:
    decode_lines: list[str] = []
    for index in reversed(range(len(out_shape))):
        decode_lines.append(f"        uint c{index} = rem % {out_shape[index]}u;")
        decode_lines.append(f"        rem = rem / {out_shape[index]}u;")

    encode_lines: list[str] = []
    for index, dim in enumerate(in_shape):
        coord = _input_coord_name(index, dim0, dim1)
        encode_lines.append(f"        in_idx = in_idx * {dim}u + {coord};")

    return (
        _SOURCE_TEMPLATE
        .replace("__DECODE_OUTPUT_COORDS__", "\n".join(decode_lines))
        .replace("__ENCODE_INPUT_INDEX__", "\n".join(encode_lines))
    )


def _input_coord_name(index: int, dim0: int, dim1: int) -> str:
    if index == dim0:
        return f"c{dim1}"
    if index == dim1:
        return f"c{dim0}"
    return f"c{index}"
