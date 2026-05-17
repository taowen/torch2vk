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
{{WEIGHT_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{X_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { {{Y_TYPE}} y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{OUT_TYPE}} output_values[]; };
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
{{WEIGHT_EXTENSION}}\
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

_RIGHT_BROADCAST_SOURCE_TEMPLATE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
{{WEIGHT_EXTENSION}}\
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

_BROADCAST_INNER_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
{{WEIGHT_EXTENSION}}\
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
        output_values[idx] = {{STORE_MUL}};
    }
}
"""

_BROADCAST_GENERAL_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
{{WEIGHT_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{X_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { {{Y_TYPE}} y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{OUT_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { {{PUSH_CONSTANTS}} } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) return;
{{COORD_SOURCE}}
    const uint x_idx = {{X_INDEX}};
    const uint y_idx = {{Y_INDEX}};
    output_values[idx] = {{STORE_MUL}};
}
"""

_SCALAR_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
{{INPUT_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{INPUT_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; float scalar; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = {{STORE_MUL}}; }
}
"""


def make_mul_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None

    if _has_scalar_arg(node):
        return _make_mul_scalar(node, out_shape, activation_dtype)

    x_shape = node_input_shape(node, 0)
    y_shape = node_input_shape(node, 1)
    if x_shape == y_shape:
        return _make_same_shape(node, activation_dtype)

    if len(x_shape) < len(y_shape):
        return _make_left_broadcast(x_shape, y_shape, out_shape, node, activation_dtype)

    if len(y_shape) < len(x_shape):
        return _make_right_broadcast(x_shape, y_shape, out_shape, node, activation_dtype)

    if len(x_shape) == len(y_shape):
        return _make_broadcast_general(x_shape, y_shape, out_shape, node, activation_dtype)

    return make_binary_same_shape(
        _same_source(activation_dtype),
        "mul_f32",
        node,
        input_dtype=activation_dtype,
        output_dtype=activation_dtype,
    )


def _same_source(activation_dtype: str) -> str:
    return _same_source_for_dtypes(activation_dtype, activation_dtype, activation_dtype)


def _same_source_for_dtypes(x_dtype: str, y_dtype: str, output_dtype: str) -> str:
    return (
        _SAME_SOURCE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(output_dtype)
        )
        .replace(
            "{{WEIGHT_EXTENSION}}",
            _mixed_float_extension_source(x_dtype, y_dtype),
        )
        .replace("{{X_TYPE}}", _glsl_type(x_dtype))
        .replace("{{Y_TYPE}}", _glsl_type(y_dtype))
        .replace("{{OUT_TYPE}}", activation_glsl_type(output_dtype))
        .replace(
            "{{STORE_MUL}}",
            activation_store(_mul_expr("x[idx]", "y[idx]", x_dtype, y_dtype), output_dtype),
        )
    )


def _make_same_shape(node: Node, activation_dtype: str) -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None
    contract_shape = shape_to_contract(out_shape)
    n_expr = flat_numel_expr(contract_shape)
    x_dtype = _input_storage_dtype(node, 0, activation_dtype)
    y_dtype = _input_storage_dtype(node, 1, activation_dtype)
    return ShaderVariant(
        name="mul_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportMulF32Program",
            shader_name="mul_f32",
            fields=(
                TensorFieldSpec(
                    "x", IOKind.INPUT, "input", TensorContract(dtype=x_dtype, shape=contract_shape)
                ),
                TensorFieldSpec(
                    "y", IOKind.INPUT, "input", TensorContract(dtype=y_dtype, shape=contract_shape)
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=contract_shape),
                ),
            ),
            push_constants=PushConstantSpec(
                size=4,
                fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_same_source_for_dtypes(x_dtype, y_dtype, activation_dtype),
        execution_requirements=activation_io_requirements(x_dtype, activation_dtype),
    )


def _has_scalar_arg(node: Node) -> bool:
    return len(node.args) >= 2 and any(not isinstance(arg, Node) for arg in node.args[:2])


def _make_mul_scalar(
    node: Node,
    out_shape: tuple[int, ...],
    activation_dtype: str,
) -> ShaderVariant:
    raw = node.args[0] if not isinstance(node.args[0], Node) else node.args[1]
    scalar_value = float(raw) if isinstance(raw, (int, float)) else 1.0
    input_dtype = _scalar_input_dtype(node, activation_dtype)
    contract_shape = shape_to_contract(out_shape)
    n_expr = flat_numel_expr(contract_shape)
    return ShaderVariant(
        name="mul_scalar",
        family="export",
        contract=ShaderContract(
            class_name="ExportMulScalarProgram",
            shader_name="mul_scalar",
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
                    TensorContract(dtype=activation_dtype, shape=contract_shape),
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
        source=_scalar_source(input_dtype, activation_dtype),
        execution_requirements=activation_io_requirements(input_dtype, activation_dtype),
    )


def _scalar_input_dtype(node: Node, activation_dtype: str) -> str:
    for arg in node.args[:2]:
        if isinstance(arg, Node):
            tensor_meta = arg.meta.get("tensor_meta")
            raw_dtype = "" if tensor_meta is None else str(tensor_meta.dtype).removeprefix("torch.")
            if raw_dtype == "int64":
                return "int64"
            if requires_float32_intermediate(arg):
                return "float32"
            return activation_dtype
    return activation_dtype


def _scalar_source(input_dtype: str, activation_dtype: str) -> str:
    return (
        _SCALAR_SOURCE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{INPUT_EXTENSION}}", _input_extension_source(input_dtype, activation_dtype))
        .replace("{{INPUT_TYPE}}", _input_glsl_type(input_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_MUL}}", activation_store("float(x[idx]) * pc.scalar", activation_dtype))
    )


def _input_glsl_type(dtype: str) -> str:
    if dtype == "int64":
        return "int64_t"
    return activation_glsl_type(dtype)


def _input_extension_source(input_dtype: str, output_dtype: str) -> str:
    if input_dtype == "int64":
        return "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n"
    if input_dtype == output_dtype or input_dtype == "float32":
        return ""
    return activation_extension_source(input_dtype)


def _make_broadcast_last(
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    node: Node,
    activation_dtype: str,
) -> ShaderVariant | None:
    out_contract = shape_to_contract(out_shape)
    y_contract = tuple(1 if d == 1 and i > 0 else out_contract[i] for i, d in enumerate(y_shape))
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
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=out_contract),
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
    x_contract = (
        (last_sym,) if len(x_shape) == 1 else shape_to_contract(x_shape, symbols=(last_sym,))
    )
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
                TensorFieldSpec(
                    "x", IOKind.INPUT, "input", TensorContract(dtype=x_dtype, shape=x_contract)
                ),
                TensorFieldSpec(
                    "y",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=out_contract),
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=out_contract),
                ),
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
        _LEFT_BROADCAST_SOURCE_TEMPLATE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{WEIGHT_EXTENSION}}", weight_extension_source(x_dtype))
        .replace("{{WEIGHT_TYPE}}", weight_glsl_type(x_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace(
            "{{STORE_MUL}}",
            activation_store(
                _mul_expr("y[idx]", "x[idx % pc.H]", activation_dtype, x_dtype), activation_dtype
            ),
        )
    )


def _make_right_broadcast(
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
    y_contract = (
        (last_sym,) if len(y_shape) == 1 else shape_to_contract(y_shape, symbols=(last_sym,))
    )
    n_expr = flat_numel_expr(out_contract)
    x_dtype = _input_storage_dtype(node, 0, activation_dtype)
    y_dtype = _input_storage_dtype(node, 1, activation_dtype)
    return ShaderVariant(
        name="mul_right_broadcast",
        family="export",
        contract=ShaderContract(
            class_name="ExportMulRightBroadcastProgram",
            shader_name="mul_right_broadcast",
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=x_dtype, shape=out_contract),
                ),
                TensorFieldSpec(
                    "y",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=y_dtype, shape=y_contract),
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=out_contract),
                ),
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
        source=_right_broadcast_source(x_dtype, y_dtype, activation_dtype),
        execution_requirements=activation_io_requirements(x_dtype, activation_dtype),
    )


def _right_broadcast_source(x_dtype: str, y_dtype: str, activation_dtype: str) -> str:
    return (
        _RIGHT_BROADCAST_SOURCE_TEMPLATE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace(
            "{{WEIGHT_EXTENSION}}",
            _mixed_float_extension_source(x_dtype, y_dtype),
        )
        .replace("{{X_TYPE}}", _glsl_type(x_dtype))
        .replace("{{Y_TYPE}}", _glsl_type(y_dtype))
        .replace("{{OUT_TYPE}}", activation_glsl_type(activation_dtype))
        .replace(
            "{{STORE_MUL}}",
            activation_store(
                _mul_expr("x[idx]", "y[idx % pc.H]", x_dtype, y_dtype), activation_dtype
            ),
        )
    )


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
        return make_binary_same_shape(
            _same_source(activation_dtype),
            "mul_f32",
            node,
            input_dtype=activation_dtype,
            output_dtype=activation_dtype,
        )
    broadcast_dim = x_broadcast_dim if x_broadcast_dim >= 0 else y_broadcast_dim
    if broadcast_dim < 0:
        return make_binary_same_shape(
            _same_source(activation_dtype),
            "mul_f32",
            node,
            input_dtype=activation_dtype,
            output_dtype=activation_dtype,
        )

    out_contract = shape_to_contract(out_shape)
    x_contract = tuple(1 if d == 1 and i > 0 else out_contract[i] for i, d in enumerate(x_shape))
    y_contract = tuple(1 if d == 1 and i > 0 else out_contract[i] for i, d in enumerate(y_shape))
    n_expr = flat_numel_expr(out_contract)

    stride = product_expr(tuple(out_contract[broadcast_dim + 1 :]))
    repeat = out_contract[broadcast_dim]
    x_dtype = _input_storage_dtype(node, 0, activation_dtype)
    y_dtype = _input_storage_dtype(node, 1, activation_dtype)

    return ShaderVariant(
        name="mul_broadcast_inner",
        family="export",
        contract=ShaderContract(
            class_name="ExportMulBroadcastInnerProgram",
            shader_name="mul_broadcast_inner",
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=x_dtype, shape=x_contract),
                ),
                TensorFieldSpec(
                    "y",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=y_dtype, shape=y_contract),
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


def _make_broadcast_general(
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    node: Node,
    activation_dtype: str,
) -> ShaderVariant | None:
    out_contract = shape_to_contract(out_shape)
    x_contract = tuple(1 if d == 1 and i > 0 else out_contract[i] for i, d in enumerate(x_shape))
    y_contract = tuple(1 if d == 1 and i > 0 else out_contract[i] for i, d in enumerate(y_shape))
    n_expr = flat_numel_expr(out_contract)
    x_dtype = _input_storage_dtype(node, 0, activation_dtype)
    y_dtype = _input_storage_dtype(node, 1, activation_dtype)
    dim_fields = tuple(
        PushConstantFieldSpec(f"D{i}", PushConstantType.UINT32, 4 * (i + 1), dim)
        for i, dim in enumerate(out_contract)
    )
    return ShaderVariant(
        name="mul_broadcast",
        family="export",
        contract=ShaderContract(
            class_name="ExportMulBroadcastProgram",
            shader_name="mul_broadcast",
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
                size=4 * (len(out_contract) + 1),
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),
                    *dim_fields,
                ),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_broadcast_general_source(
            x_shape=x_shape,
            y_shape=y_shape,
            x_dtype=x_dtype,
            y_dtype=y_dtype,
            activation_dtype=activation_dtype,
        ),
        execution_requirements=activation_io_requirements(x_dtype, activation_dtype),
    )


def _broadcast_general_source(
    *,
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    x_dtype: str,
    y_dtype: str,
    activation_dtype: str,
) -> str:
    rank = len(x_shape)
    coord_lines = ["    uint rem = idx;"]
    for i in reversed(range(rank)):
        coord_lines.append(f"    const uint c{i} = rem % pc.D{i};")
        if i > 0:
            coord_lines.append(f"    rem /= pc.D{i};")
    return (
        _BROADCAST_GENERAL_SOURCE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace(
            "{{WEIGHT_EXTENSION}}",
            _mixed_float_extension_source(x_dtype, y_dtype),
        )
        .replace("{{X_TYPE}}", _glsl_type(x_dtype))
        .replace("{{Y_TYPE}}", _glsl_type(y_dtype))
        .replace("{{OUT_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{PUSH_CONSTANTS}}", _broadcast_push_constant_source(rank))
        .replace("{{COORD_SOURCE}}", "\n".join(coord_lines))
        .replace("{{X_INDEX}}", _broadcast_flat_index_source(x_shape))
        .replace("{{Y_INDEX}}", _broadcast_flat_index_source(y_shape))
        .replace(
            "{{STORE_MUL}}",
            activation_store(_mul_expr("x[x_idx]", "y[y_idx]", x_dtype, y_dtype), activation_dtype),
        )
    )


def _broadcast_push_constant_source(rank: int) -> str:
    fields = ["uint N;"]
    fields.extend(f" uint D{i};" for i in range(rank))
    return "".join(fields)


def _broadcast_flat_index_source(shape: tuple[int, ...]) -> str:
    terms: list[str] = []
    stride_terms: list[str] = []
    for i in reversed(range(len(shape))):
        if shape[i] != 1:
            stride = " * ".join(reversed(stride_terms)) if stride_terms else "1u"
            terms.append(f"c{i} * {stride}")
        stride_terms.append(f"pc.D{i}" if shape[i] != 1 else "1u")
    if not terms:
        return "0u"
    return " + ".join(reversed(terms))


def _broadcast_last_source(*, activation_dtype: str, y_dtype: str) -> str:
    return (
        _BROADCAST_LAST_SOURCE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{WEIGHT_EXTENSION}}", _mixed_float_extension_source(activation_dtype, y_dtype))
        .replace("{{X_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{Y_TYPE}}", activation_glsl_type(y_dtype))
        .replace("{{OUT_TYPE}}", activation_glsl_type(activation_dtype))
        .replace(
            "{{STORE_MUL}}",
            activation_store(
                _mul_expr("x[idx]", "y[idx / pc.H]", activation_dtype, y_dtype), activation_dtype
            ),
        )
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
        .replace(
            "{{WEIGHT_EXTENSION}}",
            _mixed_float_extension_source(x_dtype, y_dtype),
        )
        .replace("{{X_TYPE}}", _glsl_type(x_dtype))
        .replace("{{Y_TYPE}}", _glsl_type(y_dtype))
        .replace("{{OUT_TYPE}}", activation_glsl_type(activation_dtype))
        .replace(
            "{{STORE_MUL}}",
            activation_store(_mul_expr(x_expr, y_expr, x_dtype, y_dtype), activation_dtype),
        )
    )


def _mul_expr(lhs: str, rhs: str, lhs_dtype: str, rhs_dtype: str) -> str:
    if lhs_dtype == rhs_dtype and lhs_dtype in {"float16", "float32"}:
        return f"{lhs} * {rhs}"
    return f"float({lhs}) * float({rhs})"


def _mixed_float_extension_source(*dtypes: str) -> str:
    parts: list[str] = []
    for dtype in ("float16", "bfloat16"):
        if dtype in dtypes:
            parts.append(weight_extension_source(dtype))
    return "".join(parts)


def _glsl_type(dtype: str) -> str:
    if dtype in {"float16", "float32"}:
        return activation_glsl_type(dtype)
    return weight_glsl_type(dtype)


def _input_storage_dtype(node: Node, index: int, activation_dtype: str) -> str:
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


def _is_rsqrt_rhs(node: Node) -> bool:
    rhs = node.args[1] if len(node.args) >= 2 else None
    return isinstance(rhs, Node) and str(rhs.target) == "aten.rsqrt.default"
