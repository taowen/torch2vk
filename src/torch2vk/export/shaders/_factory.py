"""Shared utilities for building ShaderVariant from FX node shapes."""

from __future__ import annotations

from torch.fx import Node

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


def shape_to_contract(shape: tuple[int, ...], symbols: tuple[str, ...] | None = None) -> tuple[Dim, ...]:
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


def make_unary_elementwise(glsl_source: str, name: str, node: Node) -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None
    contract_shape = shape_to_contract(out_shape)
    n_expr = flat_numel_expr(contract_shape)
    shader_name = f"export_{name}"
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"Export{name.title().replace('_','')}Program",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=contract_shape)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=contract_shape)),
            ),
            push_constants=PushConstantSpec(
                size=4,
                fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=glsl_source,
    )


def make_binary_same_shape(glsl_source: str, name: str, node: Node) -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None
    contract_shape = shape_to_contract(out_shape)
    n_expr = flat_numel_expr(contract_shape)
    shader_name = f"export_{name}"
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"Export{name.title().replace('_','')}Program",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=contract_shape)),
                TensorFieldSpec("y", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=contract_shape)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=contract_shape)),
            ),
            push_constants=PushConstantSpec(
                size=4,
                fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=glsl_source,
    )
