"""Shared utilities for building ShaderVariant from FX node shapes."""

from __future__ import annotations

from collections.abc import Mapping
import re

from torch.fx import Node

from torch2vk.export.dtype_policy import requires_float32_intermediate
from torch2vk.runtime.shader import (
    ExprDim,
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
from torch2vk.vulkan.types import Dim
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements

_DIM_SYMBOLS = ("B", "T", "H", "D")


def node_output_shape(node: Node) -> tuple[int, ...]:
    tm = node.meta.get("tensor_meta")
    if tm is None:
        return ()
    return tuple(int(d) for d in tm.shape)


def node_input_shape(node: Node, index: int) -> tuple[int, ...]:
    if index >= len(node.args):
        return ()
    arg = node.args[index]
    if not isinstance(arg, Node):
        return ()
    tm = arg.meta.get("tensor_meta")
    if tm is None:
        return ()
    return tuple(int(d) for d in tm.shape)


def node_input_dtype(node: Node, index: int) -> str:
    if index >= len(node.args):
        return ""
    arg = node.args[index]
    if not isinstance(arg, Node):
        return ""
    tm = arg.meta.get("tensor_meta")
    if tm is None:
        return ""
    return str(tm.dtype).removeprefix("torch.")


def node_input_storage_dtype(node: Node, index: int, activation_dtype: str) -> str:
    if index >= len(node.args):
        return activation_dtype
    arg = node.args[index]
    if not isinstance(arg, Node):
        return activation_dtype
    return node_storage_dtype(arg, activation_dtype)


def node_storage_dtype(node: Node, activation_dtype: str) -> str:
    dtype = _node_dtype(node)
    if dtype in {"int64", "int32", "uint32"}:
        return dtype
    source = _alias_source(node)
    if source.op == "placeholder" and source.name.startswith(("p_", "b_")):
        return dtype
    if requires_float32_intermediate(node):
        return "float32"
    return activation_dtype


def _alias_source(node: Node) -> Node:
    current = node
    while _is_alias_node(current):
        if not current.args or not isinstance(current.args[0], Node):
            return current
        current = current.args[0]
    return current


def _is_alias_node(node: Node) -> bool:
    return str(node.target) in {
        "aten.view.default",
        "aten.unsqueeze.default",
        "aten.reshape.default",
        "aten.contiguous.default",
        "aten._assert_tensor_metadata.default",
        "aten.to.dtype",
        "aten.to.device",
        "aten.to.dtype_layout",
        "aten.type_as.default",
    }


def _node_dtype(node: Node) -> str:
    tm = node.meta.get("tensor_meta")
    if tm is None:
        return ""
    return str(tm.dtype).removeprefix("torch.")


def weight_dtype_suffix(dtype: str) -> str:
    if dtype == "bfloat16":
        return "bf16"
    if dtype == "float16":
        return "f16"
    if dtype == "float32":
        return "f32"
    raise ValueError(f"Unsupported weight dtype for shader generation: {dtype}")


def weight_glsl_type(dtype: str) -> str:
    if dtype == "bfloat16":
        return "bfloat16_t"
    if dtype == "float16":
        return "float16_t"
    if dtype == "float32":
        return "float"
    raise ValueError(f"Unsupported weight dtype for shader generation: {dtype}")


def weight_zero_literal(dtype: str) -> str:
    if dtype == "bfloat16":
        return "bfloat16_t(0.0)"
    if dtype == "float16":
        return "float16_t(0.0)"
    if dtype == "float32":
        return "0.0"
    raise ValueError(f"Unsupported weight dtype for shader generation: {dtype}")


def weight_extension_source(dtype: str) -> str:
    if dtype == "bfloat16":
        return "#extension GL_EXT_bfloat16 : require\n"
    if dtype == "float16":
        return activation_extension_source("float16")
    if dtype == "float32":
        return ""
    raise ValueError(f"Unsupported weight dtype for shader generation: {dtype}")


def activation_glsl_type(dtype: str) -> str:
    if dtype == "float16":
        return "float16_t"
    if dtype == "float32":
        return "float"
    raise ValueError(f"Unsupported activation dtype for shader generation: {dtype}")


def activation_dtype_suffix(dtype: str) -> str:
    if dtype == "float16":
        return "f16"
    if dtype == "float32":
        return "f32"
    raise ValueError(f"Unsupported activation dtype for shader generation: {dtype}")


def activation_variant_name(base_name: str, dtype: str) -> str:
    if dtype == "float16":
        return base_name
    return f"{base_name}_act_{activation_dtype_suffix(dtype)}"


def activation_extension_source(dtype: str) -> str:
    if dtype == "float16":
        return (
            "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
            "#extension GL_EXT_shader_16bit_storage : require\n"
        )
    if dtype == "float32":
        return ""
    raise ValueError(f"Unsupported activation dtype for shader generation: {dtype}")


def activation_extension_source_for_shader(source: str, dtype: str) -> str:
    if dtype == "float32":
        return ""
    if dtype != "float16":
        raise ValueError(f"Unsupported activation dtype for shader generation: {dtype}")
    extensions: list[str] = []
    if "GL_EXT_shader_explicit_arithmetic_types_float16" not in source:
        extensions.append("#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require")
    if "GL_EXT_shader_16bit_storage" not in source:
        extensions.append("#extension GL_EXT_shader_16bit_storage : require")
    if not extensions:
        return ""
    return "\n".join(extensions) + "\n"


def activation_requirements(
    dtype: str,
    requirements: ShaderExecutionRequirements | None = None,
) -> ShaderExecutionRequirements | None:
    if dtype == "float32":
        return requirements
    if dtype != "float16":
        raise ValueError(f"Unsupported activation dtype for shader generation: {dtype}")
    if requirements is None:
        return ShaderExecutionRequirements(require_storage_buffer_16bit_access=True)
    return ShaderExecutionRequirements(
        subgroup=requirements.subgroup,
        cooperative_matrix=requirements.cooperative_matrix,
        require_integer_dot_product=requirements.require_integer_dot_product,
        require_shader_int64=requirements.require_shader_int64,
        require_buffer_device_address=requirements.require_buffer_device_address,
        require_storage_buffer_16bit_access=True,
    )


def activation_io_requirements(
    input_dtype: str,
    output_dtype: str,
    requirements: ShaderExecutionRequirements | None = None,
) -> ShaderExecutionRequirements | None:
    if "float16" in {input_dtype, output_dtype}:
        return activation_requirements("float16", requirements)
    return requirements


def activation_store(value: str, dtype: str) -> str:
    if dtype == "float16":
        return f"float16_t({value})"
    if dtype == "float32":
        return value
    raise ValueError(f"Unsupported activation dtype for shader generation: {dtype}")


def render_shader_template(source: str, replacements: Mapping[str, str]) -> str:
    rendered = source
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    placeholders = tuple(sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", rendered))))
    if placeholders:
        raise ValueError(f"shader source has unresolved placeholders: {', '.join(placeholders)}")
    return rendered


def shape_to_contract(
    shape: tuple[int, ...], symbols: tuple[str, ...] | None = None
) -> tuple[Dim, ...]:
    if symbols is None:
        symbols = _DIM_SYMBOLS if len(shape) <= 4 else tuple(f"D{i}" for i in range(len(shape)))
    result: list[Dim] = []
    for i, dim in enumerate(shape):
        if dim == 1 and i == 0:
            result.append(1)
        elif i < len(symbols):
            result.append(symbols[i])
        else:
            result.append(symbols[-1])
    return tuple(result)


def flat_numel_expr(contract_shape: tuple[Dim, ...]) -> ExprDim:
    symbols: list[str] = [s for s in contract_shape if isinstance(s, str)]
    if not symbols:
        total = 1
        for d in contract_shape:
            if isinstance(d, int):
                total *= d
        return total
    expr: ExprDim = symbols[0]
    for s in symbols[1:]:
        expr = mul(expr, s)
    return expr


def product_expr(values: tuple[ExprDim, ...]) -> ExprDim:
    expr: ExprDim = 1
    for value in values:
        if value == 1:
            continue
        if expr == 1:
            expr = value
        else:
            expr = mul(expr, value)
    return expr


def make_unary_elementwise(
    glsl_source: str,
    name: str,
    node: Node,
    *,
    input_dtype: str = "float32",
    output_dtype: str = "float32",
) -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None
    contract_shape = shape_to_contract(out_shape)
    n_expr = flat_numel_expr(contract_shape)
    shader_name = name
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"Export{name.title().replace('_', '')}Program",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=input_dtype, shape=contract_shape),
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=output_dtype, shape=contract_shape),
                ),
            ),
            push_constants=PushConstantSpec(
                size=4,
                fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=glsl_source,
        execution_requirements=activation_io_requirements(input_dtype, output_dtype),
    )


def make_binary_same_shape(
    glsl_source: str,
    name: str,
    node: Node,
    *,
    input_dtype: str = "float32",
    output_dtype: str = "float32",
) -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None
    contract_shape = shape_to_contract(out_shape)
    n_expr = flat_numel_expr(contract_shape)
    shader_name = name
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"Export{name.title().replace('_', '')}Program",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=input_dtype, shape=contract_shape),
                ),
                TensorFieldSpec(
                    "y",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=input_dtype, shape=contract_shape),
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=output_dtype, shape=contract_shape),
                ),
            ),
            push_constants=PushConstantSpec(
                size=4,
                fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=glsl_source,
        execution_requirements=activation_io_requirements(input_dtype, output_dtype),
    )
