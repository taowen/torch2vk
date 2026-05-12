from __future__ import annotations

import hashlib

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    node_input_dtype,
    node_input_shape,
    node_output_shape,
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

_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
{{INDEX_EXTENSION}}
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndexBuffer { {{INDEX_TYPE}} index_values[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
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


def make_index_select_variant(
    node: Node, activation_dtype: str = "float32"
) -> ShaderVariant | None:
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
    index_dtype = node_input_dtype(node, 2)
    if index_dtype not in {"int32", "int64"}:
        return None
    if out_shape[0] != index_shape[0] or out_shape[1] != in_shape[1]:
        return None

    shader_name = _shader_name(in_shape, index_shape, out_shape, dim, index_dtype)
    total = mul("O", "H")
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name="ExportIndexSelectF32Program",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=("N", "H")),
                ),
                TensorFieldSpec(
                    "index", IOKind.INPUT, "index", TensorContract(dtype=index_dtype, shape=("O",))
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=("O", "H")),
                ),
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
        execution_requirements=activation_requirements(
            activation_dtype, _index_execution_requirements(index_dtype)
        ),
        source=_index_source(index_dtype, activation_dtype),
    )


def _shader_name(
    in_shape: tuple[int, ...],
    index_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dim: int,
    index_dtype: str,
) -> str:
    digest = hashlib.sha1(
        repr((in_shape, index_shape, out_shape, dim, index_dtype)).encode()
    ).hexdigest()[:10]
    return f"index_select_f32_{digest}"


def _index_source(dtype: str, activation_dtype: str) -> str:
    index_type = "int64_t" if dtype == "int64" else "int"
    extension = (
        "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require"
        if dtype == "int64"
        else ""
    )
    return (
        _SOURCE.replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{INDEX_EXTENSION}}", extension)
        .replace("{{INDEX_TYPE}}", index_type)
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
    )


def _index_execution_requirements(dtype: str) -> ShaderExecutionRequirements | None:
    if dtype == "int64":
        return ShaderExecutionRequirements(require_shader_int64=True)
    return None
