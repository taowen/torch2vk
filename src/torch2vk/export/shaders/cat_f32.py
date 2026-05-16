from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
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

_SOURCE_HEADER = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
{{INPUT_BUFFERS}}layout(set = 0, binding = {{OUTPUT_BINDING}}) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N_OUT; uint OUT_STRIDE; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N_OUT) {
        uint row = idx / pc.OUT_STRIDE;
        uint col = idx % pc.OUT_STRIDE;
{{COPY_BRANCHES}}\
    }
}
"""


def make_cat_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None

    inputs = node.args[0]
    if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
        return None

    input_nodes = tuple(item for item in inputs if isinstance(item, Node))
    if len(input_nodes) != len(inputs):
        return None

    dim = int(node.args[1]) if len(node.args) > 1 and isinstance(node.args[1], int) else 0
    if dim < 0:
        dim = len(out_shape) + dim

    input_shapes: list[tuple[int, ...]] = []
    input_strides: list[int] = []
    for input_node in input_nodes:
        input_meta = input_node.meta.get("tensor_meta")
        if input_meta is None:
            return None
        input_shape = tuple(int(d) for d in input_meta.shape)
        dim_stride = 1
        for d in input_shape[dim + 1 :]:
            dim_stride *= d
        input_shapes.append(input_shape)
        input_strides.append(input_shape[dim] * dim_stride)

    out_dim_stride = 1
    for d in out_shape[dim + 1 :]:
        out_dim_stride *= d
    out_stride = out_shape[dim] * out_dim_stride

    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))
    n_out = product_expr(out_contract)
    fields = []
    for index, input_shape in enumerate(input_shapes):
        fields.append(
            TensorFieldSpec(
                f"x{index}",
                IOKind.INPUT,
                "input",
                TensorContract(
                    dtype=activation_dtype,
                    shape=tuple(f"I{index}_{dim_index}" for dim_index in range(len(input_shape))),
                ),
            )
        )
    fields.append(
        TensorFieldSpec(
            "output",
            IOKind.OUTPUT,
            "output",
            TensorContract(dtype=activation_dtype, shape=out_contract),
        )
    )

    return ShaderVariant(
        name=f"cat_{len(input_nodes)}_f32",
        family="export",
        contract=ShaderContract(
            class_name=f"ExportCat{len(input_nodes)}Program",
            shader_name=f"cat_{len(input_nodes)}_f32",
            fields=tuple(fields),
            push_constants=PushConstantSpec(
                size=8,
                fields=(
                    PushConstantFieldSpec("N_OUT", PushConstantType.UINT32, 0, n_out),
                    PushConstantFieldSpec("OUT_STRIDE", PushConstantType.UINT32, 4, out_stride),
                ),
            ),
            dispatch=(ceil_div(n_out, 256), 1, 1),
        ),
        source=_source(input_strides, activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _source(input_strides: list[int], activation_dtype: str) -> str:
    activation_type = activation_glsl_type(activation_dtype)
    input_buffers = "".join(
        f"layout(set = 0, binding = {index}) buffer restrict readonly X{index}Buffer "
        f"{{ {activation_type} x{index}[]; }};\n"
        for index in range(len(input_strides))
    )
    branches: list[str] = []
    offset = 0
    for index, stride in enumerate(input_strides):
        if stride == 0:
            continue
        condition = "if" if not branches else "else if"
        branches.append(
            f"        {condition} (col < {offset + stride}u) {{\n"
            f"            output_values[idx] = x{index}[row * {stride}u + (col - {offset}u)];\n"
            "        }\n"
        )
        offset += stride
    return _SOURCE_HEADER.replace(
        "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
    ).replace("{{ACTIVATION_TYPE}}", activation_type).replace(
        "{{INPUT_BUFFERS}}", input_buffers
    ).replace("{{OUTPUT_BINDING}}", str(len(input_strides))).replace(
        "{{COPY_BRANCHES}}", "".join(branches)
    )
