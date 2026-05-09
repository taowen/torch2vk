from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    flat_numel_expr,
    make_binary_same_shape,
    node_output_shape,
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

_SAME_SOURCE = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = x[idx] + y[idx]; }
}
"""

_SCALAR_SOURCE = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; float scalar; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = x[idx] + pc.scalar; }
}
"""


def make_add_variant(node: Node) -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None

    if len(node.args) >= 2 and not isinstance(node.args[1], Node):
        return _make_add_scalar(node, out_shape)

    return make_binary_same_shape(_SAME_SOURCE, "add_f32", node)


def _make_add_scalar(node: Node, out_shape: tuple[int, ...]) -> ShaderVariant:
    raw = node.args[1]
    scalar_value = float(raw) if isinstance(raw, (int, float)) else 0.0
    contract_shape = shape_to_contract(out_shape)
    n_expr = flat_numel_expr(contract_shape)
    return ShaderVariant(
        name="add_scalar",
        family="export",
        contract=ShaderContract(
            class_name="ExportAddScalarProgram",
            shader_name="add_scalar",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=contract_shape)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=contract_shape)),
            ),
            push_constants=PushConstantSpec(
                size=8,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),
                    PushConstantFieldSpec("scalar", PushConstantType.FLOAT32, 4, scalar_value),
                ),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_SCALAR_SOURCE,
    )
