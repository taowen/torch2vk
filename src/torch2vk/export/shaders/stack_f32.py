from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    flat_numel_expr,
    node_output_shape,
    product_expr,
    shape_to_contract,
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

_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
{{INPUT_BUFFERS}}layout(set = 0, binding = {{OUTPUT_BINDING}}) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint INNER; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const uint source = (idx / pc.INNER) % {{INPUT_COUNT}}u;
    const uint source_idx = (idx / (pc.INNER * {{INPUT_COUNT}}u)) * pc.INNER + (idx % pc.INNER);
{{COPY_BRANCHES}}\
}
"""


def make_stack_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None
    inputs = node.args[0] if node.args else None
    if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
        return None
    input_nodes = tuple(item for item in inputs if isinstance(item, Node))
    if len(input_nodes) != len(inputs):
        return None
    dim = int(node.args[1]) if len(node.args) > 1 and isinstance(node.args[1], int) else 0
    if dim < 0:
        dim += len(out_shape)
    if dim < 0 or dim >= len(out_shape):
        return None

    input_shapes: list[tuple[int, ...]] = []
    for input_node in input_nodes:
        input_meta = input_node.meta.get("tensor_meta")
        if input_meta is None:
            return None
        input_shapes.append(tuple(int(d) for d in input_meta.shape))
    if len(set(input_shapes)) != 1:
        return None

    out_contract = shape_to_contract(out_shape)
    input_contract = shape_to_contract(input_shapes[0])
    n_expr = flat_numel_expr(out_contract)
    inner_expr = product_expr(out_contract[dim + 1 :])
    fields = [
        TensorFieldSpec(
            f"x{index}",
            IOKind.INPUT,
            "input",
            TensorContract(dtype=activation_dtype, shape=input_contract),
        )
        for index in range(len(input_nodes))
    ]
    fields.append(
        TensorFieldSpec(
            "output",
            IOKind.OUTPUT,
            "output",
            TensorContract(dtype=activation_dtype, shape=out_contract),
        )
    )
    return ShaderVariant(
        name=f"stack_{len(input_nodes)}_f32",
        family="export",
        contract=ShaderContract(
            class_name=f"ExportStack{len(input_nodes)}Program",
            shader_name=f"stack_{len(input_nodes)}_f32",
            fields=tuple(fields),
            push_constants=PushConstantSpec(
                size=8,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),
                    PushConstantFieldSpec("INNER", PushConstantType.UINT32, 4, inner_expr),
                ),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_source(len(input_nodes), activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _source(input_count: int, activation_dtype: str) -> str:
    activation_type = activation_glsl_type(activation_dtype)
    input_buffers = "".join(
        f"layout(set = 0, binding = {index}) buffer restrict readonly X{index}Buffer "
        f"{{ {activation_type} x{index}[]; }};\n"
        for index in range(input_count)
    )
    branches: list[str] = []
    for index in range(input_count):
        keyword = "if" if index == 0 else "else if"
        branches.append(
            f"    {keyword} (source == {index}u) {{ output_values[idx] = x{index}[source_idx]; }}\n"
        )
    return (
        _SOURCE.replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_type)
        .replace("{{INPUT_BUFFERS}}", input_buffers)
        .replace("{{OUTPUT_BINDING}}", str(input_count))
        .replace("{{INPUT_COUNT}}", str(input_count))
        .replace("{{COPY_BRANCHES}}", "".join(branches))
    )
