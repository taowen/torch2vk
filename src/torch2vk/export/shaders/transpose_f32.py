from __future__ import annotations

import hashlib

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    node_input_shape,
    node_output_shape,
    product_expr,
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
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants {
    uint N;
__PUSH_CONSTANT_DECLS__
} pc;
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


def make_transpose_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
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

    in_contract = tuple(f"I{i}" for i in range(len(in_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))
    n_total = product_expr(out_contract)
    shader_name = _shader_name(in_shape, out_shape, dim0, dim1)
    push_fields = [PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_total)]
    offset = 4
    for name in out_contract + in_contract:
        push_fields.append(PushConstantFieldSpec(str(name), PushConstantType.UINT32, offset, name))
        offset += 4

    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name="ExportTransposeProgram",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype=activation_dtype, shape=in_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype=activation_dtype, shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=offset,
                fields=tuple(push_fields),
            ),
            dispatch=(ceil_div(n_total, 256), 1, 1),
        ),
        source=_transpose_source(len(in_shape), len(out_shape), dim0, dim1, activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _shader_name(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dim0: int,
    dim1: int,
) -> str:
    payload = f"{in_shape}->{out_shape}:{dim0}:{dim1}".encode()
    digest = hashlib.sha1(payload).hexdigest()[:10]
    return f"transpose_f32_{digest}"


def _transpose_source(
    in_rank: int,
    out_rank: int,
    dim0: int,
    dim1: int,
    activation_dtype: str,
) -> str:
    push_lines = []
    for index in range(out_rank):
        push_lines.append(f"    uint O{index};")
    for index in range(in_rank):
        push_lines.append(f"    uint I{index};")

    decode_lines: list[str] = []
    for index in reversed(range(out_rank)):
        decode_lines.append(f"        uint c{index} = rem % pc.O{index};")
        decode_lines.append(f"        rem = rem / pc.O{index};")

    encode_lines: list[str] = []
    for index in range(in_rank):
        coord = _input_coord_name(index, dim0, dim1)
        encode_lines.append(f"        in_idx = in_idx * pc.I{index} + {coord};")

    return (
        _SOURCE_TEMPLATE
        .replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("__PUSH_CONSTANT_DECLS__", "\n".join(push_lines))
        .replace("__DECODE_OUTPUT_COORDS__", "\n".join(decode_lines))
        .replace("__ENCODE_INPUT_INDEX__", "\n".join(encode_lines))
    )


def _input_coord_name(index: int, dim0: int, dim1: int) -> str:
    if index == dim0:
        return f"c{dim1}"
    if index == dim1:
        return f"c{dim0}"
    return f"c{index}"
