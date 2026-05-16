from __future__ import annotations

from torch.fx import Node

from torch2vk.export.dtype_policy import requires_float32_intermediate
from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_io_requirements,
    activation_requirements,
    activation_store,
    flat_numel_expr,
    make_binary_same_shape,
    node_input_dtype,
    node_input_shape,
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
    if (idx < pc.N) { output_values[idx] = {{STORE_ADD}}; }
}
"""

_SCALAR_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
{{INPUT_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{INPUT_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{OUTPUT_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; float scalar; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = {{STORE_ADD}}; }
}
"""

_BROADCAST_INNER_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{X_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { {{Y_TYPE}} y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{OUT_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint STRIDE; uint REPEAT; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        uint broadcast_idx = (idx / pc.STRIDE) / pc.REPEAT * pc.STRIDE + idx % pc.STRIDE;
        output_values[idx] = {{STORE_ADD}};
    }
}
"""


def make_add_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None

    if len(node.args) >= 2 and not isinstance(node.args[1], Node):
        return _make_add_scalar(node, out_shape, activation_dtype)

    x_shape = node_input_shape(node, 0)
    y_shape = node_input_shape(node, 1)
    if x_shape != y_shape and len(x_shape) == len(y_shape):
        return _make_broadcast_inner(x_shape, y_shape, out_shape, node, activation_dtype)

    return make_binary_same_shape(
        _same_source(activation_dtype),
        "add_f32",
        node,
        input_dtype=activation_dtype,
        output_dtype=activation_dtype,
    )


def _same_source(activation_dtype: str) -> str:
    return (
        _SAME_SOURCE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_ADD}}", activation_store("x[idx] + y[idx]", activation_dtype))
    )


def _make_add_scalar(
    node: Node,
    out_shape: tuple[int, ...],
    activation_dtype: str,
) -> ShaderVariant:
    raw = node.args[1]
    scalar_value = float(raw) if isinstance(raw, (int, float)) else 0.0
    input_dtype = _scalar_input_dtype(node, activation_dtype)
    output_dtype = "float32" if requires_float32_intermediate(node) else activation_dtype
    contract_shape = shape_to_contract(out_shape)
    n_expr = flat_numel_expr(contract_shape)
    return ShaderVariant(
        name="add_scalar",
        family="export",
        contract=ShaderContract(
            class_name="ExportAddScalarProgram",
            shader_name="add_scalar",
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
                size=8,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),
                    PushConstantFieldSpec("scalar", PushConstantType.FLOAT32, 4, scalar_value),
                ),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_scalar_source(input_dtype, output_dtype),
        execution_requirements=activation_io_requirements(input_dtype, output_dtype),
    )


def _scalar_input_dtype(node: Node, activation_dtype: str) -> str:
    arg = node.args[0] if node.args and isinstance(node.args[0], Node) else None
    if arg is not None and requires_float32_intermediate(arg):
        return "float32"
    return activation_dtype


def _scalar_source(input_dtype: str, output_dtype: str) -> str:
    return (
        _SCALAR_SOURCE.replace("{{ACTIVATION_EXTENSION}}", _extension_source(output_dtype))
        .replace("{{INPUT_EXTENSION}}", _input_extension_source(input_dtype, output_dtype))
        .replace("{{INPUT_TYPE}}", activation_glsl_type(input_dtype))
        .replace("{{OUTPUT_TYPE}}", activation_glsl_type(output_dtype))
        .replace("{{STORE_ADD}}", activation_store("float(x[idx]) + pc.scalar", output_dtype))
    )


def _extension_source(output_dtype: str) -> str:
    return "" if output_dtype == "float32" else activation_extension_source(output_dtype)


def _input_extension_source(input_dtype: str, output_dtype: str) -> str:
    if input_dtype == output_dtype or input_dtype == "float32":
        return ""
    return activation_extension_source(input_dtype)


def _make_broadcast_inner(
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    node: Node,
    activation_dtype: str,
) -> ShaderVariant | None:
    x_broadcast_dim = -1
    y_broadcast_dim = -1
    for i, (a, b) in enumerate(zip(x_shape, y_shape)):
        if a != b and a == 1:
            x_broadcast_dim = i
        if a != b and b == 1:
            y_broadcast_dim = i
    if x_broadcast_dim >= 0 and y_broadcast_dim >= 0:
        return None
    broadcast_dim = x_broadcast_dim if x_broadcast_dim >= 0 else y_broadcast_dim
    if broadcast_dim < 0:
        return None

    out_contract = shape_to_contract(out_shape)
    x_contract = tuple(1 if d == 1 and i > 0 else out_contract[i] for i, d in enumerate(x_shape))
    y_contract = tuple(1 if d == 1 and i > 0 else out_contract[i] for i, d in enumerate(y_shape))
    n_expr = flat_numel_expr(out_contract)
    stride = product_expr(tuple(out_contract[broadcast_dim + 1 :]))
    repeat = out_contract[broadcast_dim]
    x_dtype = _binary_input_dtype(node, 0, activation_dtype)
    y_dtype = _binary_input_dtype(node, 1, activation_dtype)

    return ShaderVariant(
        name="add_broadcast_inner",
        family="export",
        contract=ShaderContract(
            class_name="ExportAddBroadcastInnerProgram",
            shader_name="add_broadcast_inner",
            fields=(
                TensorFieldSpec(
                    "x", IOKind.INPUT, "input", TensorContract(dtype=x_dtype, shape=x_contract)
                ),
                TensorFieldSpec(
                    "y", IOKind.INPUT, "input", TensorContract(dtype=y_dtype, shape=y_contract)
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=out_contract),
                ),
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
        source=_broadcast_inner_source(
            activation_dtype=activation_dtype,
            x_dtype=x_dtype,
            y_dtype=y_dtype,
            x_is_broadcast=x_broadcast_dim >= 0,
        ),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _broadcast_inner_source(
    *,
    activation_dtype: str,
    x_dtype: str,
    y_dtype: str,
    x_is_broadcast: bool,
) -> str:
    x_expr = "x[broadcast_idx]" if x_is_broadcast else "x[idx]"
    y_expr = "y[idx]" if x_is_broadcast else "y[broadcast_idx]"
    return (
        _BROADCAST_INNER_SOURCE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{X_TYPE}}", activation_glsl_type(x_dtype))
        .replace("{{Y_TYPE}}", activation_glsl_type(y_dtype))
        .replace("{{OUT_TYPE}}", activation_glsl_type(activation_dtype))
        .replace(
            "{{STORE_ADD}}",
            activation_store(_add_expr(x_expr, y_expr, x_dtype, y_dtype), activation_dtype),
        )
    )


def _add_expr(lhs: str, rhs: str, lhs_dtype: str, rhs_dtype: str) -> str:
    if lhs_dtype == rhs_dtype and lhs_dtype in {"float16", "float32"}:
        return f"{lhs} + {rhs}"
    return f"float({lhs}) + float({rhs})"


def _binary_input_dtype(node: Node, index: int, activation_dtype: str) -> str:
    if index >= len(node.args):
        return activation_dtype
    arg = node.args[index]
    if not isinstance(arg, Node):
        return activation_dtype
    dtype = node_input_dtype(node, index) or activation_dtype
    if dtype in {"int64", "int32", "uint32"}:
        return dtype
    if arg.op == "placeholder":
        return dtype if arg.name.startswith("p_") else activation_dtype
    return "float32" if requires_float32_intermediate(arg) else activation_dtype
