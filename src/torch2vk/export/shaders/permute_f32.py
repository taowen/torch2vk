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
    if (idx >= pc.N) return;
    uint rem = idx;
__DECODE_OUTPUT_COORDS__
    uint in_idx = 0u;
__ENCODE_INPUT_INDEX__
    output_values[idx] = x[in_idx];
}
"""


def make_permute_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    input_shape = node_input_shape(node, 0)
    output_shape = node_output_shape(node)
    if not input_shape or not output_shape or len(input_shape) != len(output_shape):
        return None
    if len(node.args) < 2 or not isinstance(node.args[1], (list, tuple)):
        return None
    dims_list: list[int] = []
    for dim in node.args[1]:
        if not isinstance(dim, int):
            return None
        dims_list.append(dim)
    dims = tuple(dims_list)
    rank = len(input_shape)
    dims = tuple(dim + rank if dim < 0 else dim for dim in dims)
    if sorted(dims) != list(range(rank)):
        return None

    input_contract = tuple(f"I{i}" for i in range(rank))
    output_contract = tuple(f"O{i}" for i in range(rank))
    n_expr = product_expr(output_contract)
    shader_name = _shader_name(input_shape, output_shape, dims)
    push_fields = [PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr)]
    offset = 4
    for name in output_contract + input_contract:
        push_fields.append(PushConstantFieldSpec(name, PushConstantType.UINT32, offset, name))
        offset += 4

    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name="ExportPermuteProgram",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=input_contract),
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=output_contract),
                ),
            ),
            push_constants=PushConstantSpec(size=offset, fields=tuple(push_fields)),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_source(rank, dims, activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _shader_name(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    dims: tuple[int, ...],
) -> str:
    payload = f"{input_shape}->{output_shape}:{dims}".encode()
    digest = hashlib.sha1(payload).hexdigest()[:10]
    return f"permute_f32_{digest}"


def _source(rank: int, dims: tuple[int, ...], activation_dtype: str) -> str:
    push_lines = []
    for index in range(rank):
        push_lines.append(f"    uint O{index};")
    for index in range(rank):
        push_lines.append(f"    uint I{index};")

    decode_lines: list[str] = []
    for index in reversed(range(rank)):
        decode_lines.append(f"    uint c{index} = rem % pc.O{index};")
        decode_lines.append(f"    rem = rem / pc.O{index};")

    inverse = [0] * rank
    for output_index, input_index in enumerate(dims):
        inverse[input_index] = output_index

    encode_lines: list[str] = []
    for input_index in range(rank):
        encode_lines.append(f"    in_idx = in_idx * pc.I{input_index} + c{inverse[input_index]};")

    return (
        _SOURCE_TEMPLATE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("__PUSH_CONSTANT_DECLS__", "\n".join(push_lines))
        .replace("__DECODE_OUTPUT_COORDS__", "\n".join(decode_lines))
        .replace("__ENCODE_INPUT_INDEX__", "\n".join(encode_lines))
    )
