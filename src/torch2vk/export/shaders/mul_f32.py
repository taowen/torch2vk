from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    activation_store,
    flat_numel_expr,
    make_binary_same_shape,
    node_input_dtype,
    node_input_shape,
    node_output_shape,
    shape_to_contract,
    weight_dtype_suffix,
    weight_extension_source,
    weight_glsl_type,
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
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { {{ACTIVATION_TYPE}} y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = {{STORE_MUL}}; }
}
"""

_BROADCAST_LAST_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{X_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { {{Y_TYPE}} y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{OUT_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint H; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = {{STORE_MUL}}; }
}
"""

_LEFT_BROADCAST_SOURCE_TEMPLATE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
{{WEIGHT_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{WEIGHT_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { {{ACTIVATION_TYPE}} y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint H; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = {{STORE_MUL}}; }
}
"""

_BROADCAST_INNER_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { {{ACTIVATION_TYPE}} y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint STRIDE; uint REPEAT; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        uint y_idx = (idx / pc.STRIDE) / pc.REPEAT * pc.STRIDE + idx % pc.STRIDE;
        output_values[idx] = {{STORE_MUL}};
    }
}
"""


def make_mul_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    y_shape = node_input_shape(node, 1)
    out_shape = node_output_shape(node)
    if not out_shape:
        return None

    if x_shape == y_shape:
        return make_binary_same_shape(
            _same_source(activation_dtype),
            "mul_f32",
            node,
            input_dtype=activation_dtype,
            output_dtype=activation_dtype,
        )

    if len(x_shape) < len(y_shape):
        return _make_left_broadcast(x_shape, y_shape, out_shape, node, activation_dtype)

    if len(x_shape) == len(y_shape) and y_shape[-1] == 1:
        return _make_broadcast_last(x_shape, y_shape, out_shape, node, activation_dtype)

    if len(x_shape) == len(y_shape):
        return _make_broadcast_inner(x_shape, y_shape, out_shape, node, activation_dtype)

    return make_binary_same_shape(
        _same_source(activation_dtype),
        "mul_f32",
        node,
        input_dtype=activation_dtype,
        output_dtype=activation_dtype,
    )


def _same_source(activation_dtype: str) -> str:
    return (
        _SAME_SOURCE
        .replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_MUL}}", activation_store("x[idx] * y[idx]", activation_dtype))
    )


def _make_broadcast_last(
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    node: Node,
    activation_dtype: str,
) -> ShaderVariant:
    out_contract = shape_to_contract(out_shape)
    y_contract = tuple(
        1 if d == 1 and i > 0 else out_contract[i]
        for i, d in enumerate(y_shape)
    )
    n_expr = flat_numel_expr(out_contract)
    last_sym = out_contract[-1]
    y_dtype = "float32" if _is_rsqrt_rhs(node) else activation_dtype
    return ShaderVariant(
        name="mul_broadcast_last",
        family="export",
        contract=ShaderContract(
            class_name="ExportMulBroadcastLastProgram",
            shader_name="mul_broadcast_last",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype=activation_dtype, shape=out_contract)),
                TensorFieldSpec("y", IOKind.INPUT, "input", TensorContract(dtype=y_dtype, shape=y_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype=activation_dtype, shape=out_contract)),
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
        source=_broadcast_last_source(activation_dtype=activation_dtype, y_dtype=y_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _make_left_broadcast(
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    node: Node,
    activation_dtype: str,
) -> ShaderVariant:
    out_contract = shape_to_contract(out_shape)
    last_sym = out_contract[-1] if out_contract else "D"
    if not isinstance(last_sym, str):
        last_sym = "D"
    x_contract = (last_sym,) if len(x_shape) == 1 else shape_to_contract(x_shape, symbols=(last_sym,))
    n_expr = flat_numel_expr(out_contract)
    h_sym = last_sym
    x_dtype = node_input_dtype(node, 0)
    x_suffix = weight_dtype_suffix(x_dtype)
    shader_name = f"mul_left_broadcast_{x_suffix}x_f32"
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"ExportMulLeftBroadcast{x_suffix.title()}XProgram",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype=x_dtype, shape=x_contract)),
                TensorFieldSpec("y", IOKind.INPUT, "input", TensorContract(dtype=activation_dtype, shape=out_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype=activation_dtype, shape=out_contract)),
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
        source=_left_broadcast_source(x_dtype, activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _left_broadcast_source(x_dtype: str, activation_dtype: str) -> str:
    return (
        _LEFT_BROADCAST_SOURCE_TEMPLATE
        .replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{WEIGHT_EXTENSION}}", weight_extension_source(x_dtype))
        .replace("{{WEIGHT_TYPE}}", weight_glsl_type(x_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_MUL}}", activation_store(_mul_expr("y[idx]", "x[idx % pc.H]", activation_dtype, x_dtype), activation_dtype))
    )


def _make_broadcast_inner(
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    node: Node,
    activation_dtype: str,
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
        name="mul_broadcast_inner",
        family="export",
        contract=ShaderContract(
            class_name="ExportMulBroadcastInnerProgram",
            shader_name="mul_broadcast_inner",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype=activation_dtype, shape=out_contract)),
                TensorFieldSpec("y", IOKind.INPUT, "input", TensorContract(dtype=activation_dtype, shape=y_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype=activation_dtype, shape=out_contract)),
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
        source=_broadcast_inner_source(activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _broadcast_last_source(*, activation_dtype: str, y_dtype: str) -> str:
    return (
        _BROADCAST_LAST_SOURCE
        .replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{X_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{Y_TYPE}}", activation_glsl_type(y_dtype))
        .replace("{{OUT_TYPE}}", activation_glsl_type(activation_dtype))
        .replace(
            "{{STORE_MUL}}",
            activation_store(_mul_expr("x[idx]", "y[idx / pc.H]", activation_dtype, y_dtype), activation_dtype),
        )
    )


def _broadcast_inner_source(activation_dtype: str) -> str:
    return (
        _BROADCAST_INNER_SOURCE
        .replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_MUL}}", activation_store("x[idx] * y[y_idx]", activation_dtype))
    )


def _mul_expr(lhs: str, rhs: str, lhs_dtype: str, rhs_dtype: str) -> str:
    if lhs_dtype == rhs_dtype and lhs_dtype in {"float16", "float32"}:
        return f"{lhs} * {rhs}"
    return f"float({lhs}) * float({rhs})"


def _is_rsqrt_rhs(node: Node) -> bool:
    rhs = node.args[1] if len(node.args) >= 2 else None
    return isinstance(rhs, Node) and str(rhs.target) == "aten.rsqrt.default"
