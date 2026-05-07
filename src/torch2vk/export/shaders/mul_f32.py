from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    flat_numel_expr,
    make_binary_same_shape,
    node_input_shape,
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
    if (idx < pc.N) { output_values[idx] = x[idx] * y[idx]; }
}
"""

_BROADCAST_LAST_SOURCE = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint H; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = x[idx] * y[idx / pc.H]; }
}
"""

_LEFT_BROADCAST_SOURCE = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint H; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = x[idx % pc.H] * y[idx]; }
}
"""

_BROADCAST_INNER_SOURCE = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint STRIDE; uint REPEAT; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        uint y_idx = (idx / pc.STRIDE) / pc.REPEAT * pc.STRIDE + idx % pc.STRIDE;
        output_values[idx] = x[idx] * y[y_idx];
    }
}
"""


def make_mul_variant(node: Node) -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    y_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not out_shape:
        return None

    if x_shape == y_shape:
        return make_binary_same_shape(_SAME_SOURCE, "mul_f32", node)

    if len(x_shape) < len(y_shape):
        return _make_left_broadcast(x_shape, y_shape, out_shape, node)

    if len(x_shape) == len(y_shape) and y_shape[-1] == 1:
        return _make_broadcast_last(x_shape, y_shape, out_shape, node)

    if len(x_shape) == len(y_shape):
        return _make_broadcast_inner(x_shape, y_shape, out_shape, node)

    return make_binary_same_shape(_SAME_SOURCE, "mul_f32", node)


def _make_broadcast_last(
    x_shape: tuple[int, ...], y_shape: tuple[int, ...], out_shape: tuple[int, ...], node: Node
) -> ShaderVariant:
    out_contract = shape_to_contract(out_shape)
    y_contract = tuple(
        1 if d == 1 and i > 0 else out_contract[i]
        for i, d in enumerate(y_shape)
    )
    n_expr = flat_numel_expr(out_contract)
    last_sym = out_contract[-1]
    return ShaderVariant(
        name="export_mul_broadcast_last",
        family="export",
        contract=ShaderContract(
            class_name="ExportMulBroadcastLastProgram",
            shader_name="export_mul_broadcast_last",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=out_contract)),
                TensorFieldSpec("y", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=y_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=8,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),
                    PushConstantFieldSpec("H", PushConstantType.UINT32, 4, last_sym),
                ),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_BROADCAST_LAST_SOURCE,
    )


def _make_left_broadcast(
    x_shape: tuple[int, ...], y_shape: tuple[int, ...], out_shape: tuple[int, ...], node: Node
) -> ShaderVariant:
    out_contract = shape_to_contract(out_shape)
    last_sym = out_contract[-1] if out_contract else "D"
    if not isinstance(last_sym, str):
        last_sym = "D"
    x_contract = (last_sym,) if len(x_shape) == 1 else shape_to_contract(x_shape, symbols=(last_sym,))
    n_expr = flat_numel_expr(out_contract)
    h_sym = last_sym
    return ShaderVariant(
        name="export_mul_left_broadcast",
        family="export",
        contract=ShaderContract(
            class_name="ExportMulLeftBroadcastProgram",
            shader_name="export_mul_left_broadcast",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=x_contract)),
                TensorFieldSpec("y", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=out_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=8,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),
                    PushConstantFieldSpec("H", PushConstantType.UINT32, 4, h_sym),
                ),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_LEFT_BROADCAST_SOURCE,
    )


def _make_broadcast_inner(
    x_shape: tuple[int, ...], y_shape: tuple[int, ...], out_shape: tuple[int, ...], node: Node
) -> ShaderVariant:
    broadcast_dim = -1
    for i, (a, b) in enumerate(zip(x_shape, y_shape)):
        if a != b and b == 1:
            broadcast_dim = i
            break

    out_contract = shape_to_contract(out_shape)
    y_contract = tuple(
        1 if d == 1 and i > 0 else out_contract[i]
        for i, d in enumerate(y_shape)
    )
    n_expr = flat_numel_expr(out_contract)

    stride = 1
    for d in out_shape[broadcast_dim + 1:]:
        stride *= d
    repeat = out_shape[broadcast_dim]

    return ShaderVariant(
        name="export_mul_broadcast_inner",
        family="export",
        contract=ShaderContract(
            class_name="ExportMulBroadcastInnerProgram",
            shader_name="export_mul_broadcast_inner",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=out_contract)),
                TensorFieldSpec("y", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=y_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=12,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),
                    PushConstantFieldSpec("STRIDE", PushConstantType.UINT32, 4, stride),
                    PushConstantFieldSpec("REPEAT", PushConstantType.UINT32, 8, repeat),
                ),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_BROADCAST_INNER_SOURCE,
    )
