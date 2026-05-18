from __future__ import annotations

from dataclasses import dataclass

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    node_input_shape,
    node_output_shape,
    node_storage_dtype,
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
    mul,
)
from torch2vk.vulkan.types import dtype_nbytes

_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N_OUT; uint IN_STRIDE; uint OUT_STRIDE; uint OFFSET; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N_OUT) {
        uint row = idx / pc.OUT_STRIDE;
        uint col = idx % pc.OUT_STRIDE;
        output_values[idx] = x[row * pc.IN_STRIDE + pc.OFFSET + col];
    }
}
"""

_UNBIND_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint SELECT_DIM; uint INNER; uint SELECTED; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const uint outer = idx / pc.INNER;
    const uint inner = idx - outer * pc.INNER;
    output_values[idx] = x[(outer * pc.SELECT_DIM + pc.SELECTED) * pc.INNER + inner];
}
"""


@dataclass(frozen=True, slots=True)
class TupleGetitemAlias:
    src: str
    byte_offset: int
    nbytes: int


def tuple_getitem_alias(
    node: Node,
    activation_dtype: str = "float32",
) -> TupleGetitemAlias | None:
    source_node, tuple_index = _tuple_getitem_args(node)
    if source_node is None or tuple_index is None:
        return None
    target = str(source_node.target)
    if target == "aten.chunk.default":
        return _chunk_getitem_alias(
            node,
            source_node=source_node,
            tuple_index=tuple_index,
            activation_dtype=activation_dtype,
        )
    if target == "aten.split_with_sizes.default":
        return _split_getitem_alias(
            node,
            source_node=source_node,
            tuple_index=tuple_index,
            activation_dtype=activation_dtype,
        )
    return None


def make_tuple_getitem_variant(
    node: Node,
    activation_dtype: str = "float32",
) -> ShaderVariant | None:
    if tuple_getitem_alias(node, activation_dtype) is not None:
        return None
    source_node, tuple_index = _tuple_getitem_args(node)
    if source_node is None or tuple_index is None:
        return None
    target = str(source_node.target)
    if target == "aten.chunk.default":
        return _make_chunk_getitem_variant(
            node,
            source_node=source_node,
            tuple_index=tuple_index,
            activation_dtype=activation_dtype,
        )
    if target == "aten.split_with_sizes.default":
        return _make_split_getitem_variant(
            node,
            source_node=source_node,
            tuple_index=tuple_index,
            activation_dtype=activation_dtype,
        )
    if target == "aten.unbind.int":
        return _make_unbind_getitem_variant(
            node,
            source_node=source_node,
            tuple_index=tuple_index,
            activation_dtype=activation_dtype,
        )
    return None


def _tuple_getitem_args(node: Node) -> tuple[Node | None, int | None]:
    if str(node.target) != "<built-in function getitem>" or len(node.args) != 2:
        return None, None
    source_node = node.args[0]
    tuple_index = node.args[1]
    if not isinstance(source_node, Node) or not isinstance(tuple_index, int):
        return None, None
    return source_node, tuple_index


def _make_chunk_getitem_variant(
    node: Node,
    *,
    source_node: Node,
    tuple_index: int,
    activation_dtype: str,
) -> ShaderVariant | None:
    input_shape = node_input_shape(source_node, 0)
    output_shape = node_output_shape(node)
    if not input_shape or not output_shape:
        return None
    chunks = source_node.args[1] if len(source_node.args) > 1 else None
    dim = source_node.args[2] if len(source_node.args) > 2 else 0
    if not isinstance(chunks, int) or not isinstance(dim, int):
        return None
    if dim < 0:
        dim += len(input_shape)
    if tuple_index < 0 or tuple_index >= chunks:
        return None
    full = input_shape[dim]
    base, extra = divmod(full, chunks)
    sizes = tuple(base + 1 if i < extra else base for i in range(chunks))
    return _make_tuple_slice_variant(
        node,
        source_node=source_node,
        dim=dim,
        start=sum(sizes[:tuple_index]),
        activation_dtype=activation_dtype,
    )


def _chunk_getitem_alias(
    node: Node,
    *,
    source_node: Node,
    tuple_index: int,
    activation_dtype: str,
) -> TupleGetitemAlias | None:
    input_shape = node_input_shape(source_node, 0)
    if not input_shape:
        return None
    chunks = source_node.args[1] if len(source_node.args) > 1 else None
    dim = source_node.args[2] if len(source_node.args) > 2 else 0
    if not isinstance(chunks, int) or not isinstance(dim, int):
        return None
    if dim < 0:
        dim += len(input_shape)
    if tuple_index < 0 or tuple_index >= chunks or dim < 0 or dim >= len(input_shape):
        return None
    full = input_shape[dim]
    base, extra = divmod(full, chunks)
    sizes = tuple(base + 1 if i < extra else base for i in range(chunks))
    return _tuple_slice_alias(
        node,
        source_node=source_node,
        dim=dim,
        start=sum(sizes[:tuple_index]),
        activation_dtype=activation_dtype,
    )


def _make_split_getitem_variant(
    node: Node,
    *,
    source_node: Node,
    tuple_index: int,
    activation_dtype: str,
) -> ShaderVariant | None:
    input_shape = node_input_shape(source_node, 0)
    output_shape = node_output_shape(node)
    if not input_shape or not output_shape:
        return None
    split_sizes = source_node.args[1] if len(source_node.args) > 1 else None
    dim = source_node.args[2] if len(source_node.args) > 2 else 0
    if not isinstance(split_sizes, (list, tuple)) or not isinstance(dim, int):
        return None
    sizes_list: list[int] = []
    for size in split_sizes:
        if not isinstance(size, int):
            return None
        sizes_list.append(size)
    sizes = tuple(sizes_list)
    if dim < 0:
        dim += len(input_shape)
    if tuple_index < 0 or tuple_index >= len(sizes):
        return None
    return _make_tuple_slice_variant(
        node,
        source_node=source_node,
        dim=dim,
        start=sum(sizes[:tuple_index]),
        activation_dtype=activation_dtype,
    )


def _split_getitem_alias(
    node: Node,
    *,
    source_node: Node,
    tuple_index: int,
    activation_dtype: str,
) -> TupleGetitemAlias | None:
    input_shape = node_input_shape(source_node, 0)
    if not input_shape:
        return None
    split_sizes = source_node.args[1] if len(source_node.args) > 1 else None
    dim = source_node.args[2] if len(source_node.args) > 2 else 0
    if not isinstance(split_sizes, (list, tuple)) or not isinstance(dim, int):
        return None
    sizes: list[int] = []
    for size in split_sizes:
        if not isinstance(size, int):
            return None
        sizes.append(size)
    if dim < 0:
        dim += len(input_shape)
    if tuple_index < 0 or tuple_index >= len(sizes) or dim < 0 or dim >= len(input_shape):
        return None
    return _tuple_slice_alias(
        node,
        source_node=source_node,
        dim=dim,
        start=sum(sizes[:tuple_index]),
        activation_dtype=activation_dtype,
    )


def _tuple_slice_alias(
    node: Node,
    *,
    source_node: Node,
    dim: int,
    start: int,
    activation_dtype: str,
) -> TupleGetitemAlias | None:
    input_node = source_node.args[0] if source_node.args else None
    if not isinstance(input_node, Node):
        return None
    input_shape = node_input_shape(source_node, 0)
    output_shape = node_output_shape(node)
    if not input_shape or not output_shape:
        return None
    if dim < 0 or dim >= len(input_shape):
        return None
    outer = 1
    for size in input_shape[:dim]:
        outer *= size
    if outer != 1:
        return None
    inner = 1
    for size in input_shape[dim + 1 :]:
        inner *= size
    output_elements = 1
    for size in output_shape:
        output_elements *= size
    element_nbytes = dtype_nbytes(node_storage_dtype(input_node, activation_dtype))
    return TupleGetitemAlias(
        src=input_node.name,
        byte_offset=start * inner * element_nbytes,
        nbytes=output_elements * element_nbytes,
    )


def _make_tuple_slice_variant(
    node: Node,
    *,
    source_node: Node,
    dim: int,
    start: int,
    activation_dtype: str,
) -> ShaderVariant | None:
    input_node = source_node.args[0] if source_node.args else None
    if not isinstance(input_node, Node):
        return None
    in_shape = node_input_shape(source_node, 0)
    out_shape = node_output_shape(node)
    if dim < 0 or dim >= len(in_shape):
        return None
    node.meta["torch2vk_shader_inputs"] = (input_node.name,)
    in_contract = tuple(f"I{i}" for i in range(len(in_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))
    inner_stride = product_expr(in_contract[dim + 1 :])
    out_inner_stride = product_expr(out_contract[dim + 1 :])
    in_stride_val = mul(in_contract[dim], inner_stride)
    out_stride_val = mul(out_contract[dim], out_inner_stride)
    offset = 0 if start == 0 else mul(start, inner_stride)
    n_out = product_expr(out_contract)
    return ShaderVariant(
        name="tuple_getitem_slice_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportTupleGetitemSliceProgram",
            shader_name="tuple_getitem_slice_f32",
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=in_contract),
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=out_contract),
                ),
            ),
            push_constants=PushConstantSpec(
                size=16,
                fields=(
                    PushConstantFieldSpec("N_OUT", PushConstantType.UINT32, 0, n_out),
                    PushConstantFieldSpec("IN_STRIDE", PushConstantType.UINT32, 4, in_stride_val),
                    PushConstantFieldSpec("OUT_STRIDE", PushConstantType.UINT32, 8, out_stride_val),
                    PushConstantFieldSpec("OFFSET", PushConstantType.UINT32, 12, offset),
                ),
            ),
            dispatch=(ceil_div(n_out, 256), 1, 1),
        ),
        source=_source(activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _make_unbind_getitem_variant(
    node: Node,
    *,
    source_node: Node,
    tuple_index: int,
    activation_dtype: str,
) -> ShaderVariant | None:
    input_node = source_node.args[0] if source_node.args else None
    if not isinstance(input_node, Node):
        return None
    input_shape = node_input_shape(source_node, 0)
    output_shape = node_output_shape(node)
    dim = source_node.args[1] if len(source_node.args) > 1 else 0
    if not input_shape or not output_shape or not isinstance(dim, int):
        return None
    if dim < 0:
        dim += len(input_shape)
    if dim < 0 or dim >= len(input_shape):
        return None
    if tuple_index < 0:
        tuple_index += input_shape[dim]
    if tuple_index < 0 or tuple_index >= input_shape[dim]:
        return None

    node.meta["torch2vk_shader_inputs"] = (input_node.name,)
    input_contract = tuple(f"I{i}" for i in range(len(input_shape)))
    output_contract = tuple(f"O{i}" for i in range(len(output_shape)))
    n_out = product_expr(output_contract)
    inner = product_expr(input_contract[dim + 1 :])
    return ShaderVariant(
        name="tuple_getitem_unbind_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportTupleGetitemUnbindProgram",
            shader_name="tuple_getitem_unbind_f32",
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
            push_constants=PushConstantSpec(
                size=16,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_out),
                    PushConstantFieldSpec(
                        "SELECT_DIM", PushConstantType.UINT32, 4, input_contract[dim]
                    ),
                    PushConstantFieldSpec("INNER", PushConstantType.UINT32, 8, inner),
                    PushConstantFieldSpec("SELECTED", PushConstantType.UINT32, 12, tuple_index),
                ),
            ),
            dispatch=(ceil_div(n_out, 256), 1, 1),
        ),
        source=_unbind_source(activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _source(activation_dtype: str) -> str:
    return _SOURCE.replace(
        "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
    ).replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))


def _unbind_source(activation_dtype: str) -> str:
    return _UNBIND_SOURCE.replace(
        "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
    ).replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
