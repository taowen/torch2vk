from __future__ import annotations

from torch.fx import Node

from torch2vk.export.dtype_policy import requires_float32_intermediate
from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    node_input_dtype,
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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements

_SOURCE = """\
#version 450
{{EXTENSION}}
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{DATA_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{DATA_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint select_dim; uint inner; uint selected; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) return;
    const uint outer = idx / pc.inner;
    const uint inner = idx - outer * pc.inner;
    output_values[idx] = x[(outer * pc.select_dim + pc.selected) * pc.inner + inner];
}
"""


def make_select_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    input_shape = node_input_shape(node, 0)
    output_shape = node_output_shape(node)
    raw_dtype = node_input_dtype(node, 0)
    if not input_shape or not output_shape or raw_dtype not in {
        "float16",
        "float32",
        "int32",
        "int64",
        "uint32",
    }:
        return None
    if len(node.args) < 3 or not isinstance(node.args[1], int) or not isinstance(node.args[2], int):
        return None
    dim = int(node.args[1])
    selected = int(node.args[2])
    if dim < 0:
        dim += len(input_shape)
    if not (0 <= dim < len(input_shape)):
        return None
    if selected < 0:
        selected += input_shape[dim]
    if not (0 <= selected < input_shape[dim]):
        return None

    dtype = _storage_dtype(node, raw_dtype, activation_dtype)
    in_contract = tuple(f"I{i}" for i in range(len(input_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(output_shape)))
    n_expr = product_expr(out_contract)
    inner_expr = product_expr(in_contract[dim + 1 :])
    return ShaderVariant(
        name=f"select_{dtype}",
        family="export",
        contract=ShaderContract(
            class_name="ExportSelectProgram",
            shader_name=f"select_{dtype}",
            fields=(
                TensorFieldSpec(
                    "x", IOKind.INPUT, "input", TensorContract(dtype=dtype, shape=in_contract)
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=dtype, shape=out_contract),
                ),
            ),
            push_constants=PushConstantSpec(
                size=16,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),
                    PushConstantFieldSpec(
                        "select_dim", PushConstantType.UINT32, 4, in_contract[dim]
                    ),
                    PushConstantFieldSpec("inner", PushConstantType.UINT32, 8, inner_expr),
                    PushConstantFieldSpec("selected", PushConstantType.UINT32, 12, selected),
                ),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_source(dtype),
        execution_requirements=_requirements(dtype),
    )


def _source(dtype: str) -> str:
    if dtype == "int64":
        data_type = "int64_t"
        extension = "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n"
    elif dtype == "int32":
        data_type = "int"
        extension = ""
    elif dtype == "uint32":
        data_type = "uint"
        extension = ""
    elif dtype == "float16":
        data_type = activation_glsl_type(dtype)
        extension = activation_extension_source(dtype)
    elif dtype == "float32":
        data_type = "float"
        extension = ""
    else:
        raise ValueError(f"Unsupported select dtype: {dtype}")
    return _SOURCE.replace("{{EXTENSION}}", extension).replace("{{DATA_TYPE}}", data_type)


def _requirements(dtype: str) -> ShaderExecutionRequirements | None:
    if dtype == "int64":
        return ShaderExecutionRequirements(require_shader_int64=True)
    if dtype in {"float16", "float32"}:
        return activation_requirements(dtype)
    return None


def _storage_dtype(node: Node, raw_dtype: str, activation_dtype: str) -> str:
    if raw_dtype in {"int32", "int64", "uint32"}:
        return raw_dtype
    arg = node.args[0] if node.args and isinstance(node.args[0], Node) else None
    if arg is not None and requires_float32_intermediate(arg):
        return "float32"
    return activation_dtype
