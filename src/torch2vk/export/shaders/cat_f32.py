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
    add,
    ceil_div,
    mul,
)

_SOURCE_HEADER = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
{{INPUT_BUFFERS}}layout(set = 0, binding = {{OUTPUT_BINDING}}) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N_OUT; uint OUT_STRIDE; {{PUSH_CONSTANT_FIELDS}} } pc;
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

    input_contracts: list[tuple[str, ...]] = []
    input_strides = []
    for input_node in input_nodes:
        input_meta = input_node.meta.get("tensor_meta")
        if input_meta is None:
            return None
        input_shape = tuple(int(d) for d in input_meta.shape)
        input_contract = tuple(f"I{len(input_contracts)}_{dim_index}" for dim_index in range(len(input_shape)))
        input_contracts.append(input_contract)
        dim_stride = product_expr(input_contract[dim + 1 :])
        input_strides.append(mul(input_contract[dim], dim_stride))

    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))
    out_dim_stride = product_expr(out_contract[dim + 1 :])
    out_stride = mul(out_contract[dim], out_dim_stride)
    n_out = product_expr(out_contract)
    fields = []
    for index, input_contract in enumerate(input_contracts):
        fields.append(
            TensorFieldSpec(
                f"x{index}",
                IOKind.INPUT,
                "input",
                TensorContract(
                    dtype=activation_dtype,
                    shape=input_contract,
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

    push_constant_fields = [
        PushConstantFieldSpec("N_OUT", PushConstantType.UINT32, 0, n_out),
        PushConstantFieldSpec("OUT_STRIDE", PushConstantType.UINT32, 4, out_stride),
    ]
    segment_end = 0
    for index, stride in enumerate(input_strides):
        segment_end = stride if segment_end == 0 else add(segment_end, stride)
        push_constant_fields.append(
            PushConstantFieldSpec(f"END{index}", PushConstantType.UINT32, 8 + index * 8, segment_end)
        )
        push_constant_fields.append(
            PushConstantFieldSpec(f"STRIDE{index}", PushConstantType.UINT32, 12 + index * 8, stride)
        )

    return ShaderVariant(
        name=f"cat_{len(input_nodes)}_f32",
        family="export",
        contract=ShaderContract(
            class_name=f"ExportCat{len(input_nodes)}Program",
            shader_name=f"cat_{len(input_nodes)}_f32",
            fields=tuple(fields),
            push_constants=PushConstantSpec(
                size=8 + 8 * len(input_strides),
                fields=tuple(push_constant_fields),
            ),
            dispatch=(ceil_div(n_out, 256), 1, 1),
        ),
        source=_source(len(input_strides), activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _source(input_count: int, activation_dtype: str) -> str:
    activation_type = activation_glsl_type(activation_dtype)
    input_buffers = "".join(
        f"layout(set = 0, binding = {index}) buffer restrict readonly X{index}Buffer "
        f"{{ {activation_type} x{index}[]; }};\n"
        for index in range(input_count)
    )
    push_constant_fields = "".join(
        f"uint END{index}; uint STRIDE{index}; " for index in range(input_count)
    )
    branches: list[str] = []
    for index in range(input_count):
        condition = "if" if not branches else "else if"
        offset = "0u" if index == 0 else f"pc.END{index - 1}"
        branches.append(
            f"        {condition} (col < pc.END{index}) {{\n"
            f"            output_values[idx] = x{index}[row * pc.STRIDE{index} + (col - {offset})];\n"
            "        }\n"
        )
    return _SOURCE_HEADER.replace(
        "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
    ).replace("{{ACTIVATION_TYPE}}", activation_type).replace(
        "{{INPUT_BUFFERS}}", input_buffers
    ).replace("{{PUSH_CONSTANT_FIELDS}}", push_constant_fields).replace(
        " ;", ";"
    ).replace("{{OUTPUT_BINDING}}", str(input_count)).replace(
        "{{COPY_BRANCHES}}", "".join(branches)
    )
