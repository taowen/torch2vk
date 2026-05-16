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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements

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
    if (idx < pc.N) { output_values[idx] = {{STORE_DIV}}; }
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
    if (idx < pc.N) { output_values[idx] = {{STORE_DIV}}; }
}
"""


def make_div_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None
    if len(node.args) >= 2 and not isinstance(node.args[1], Node):
        return _make_div_scalar(node, out_shape, activation_dtype)
    return make_binary_same_shape(
        _same_source(activation_dtype),
        "div_f32",
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
        .replace("{{STORE_DIV}}", activation_store("float(x[idx]) / float(y[idx])", activation_dtype))
    )


def _make_div_scalar(
    node: Node,
    out_shape: tuple[int, ...],
    activation_dtype: str,
) -> ShaderVariant:
    raw = node.args[1]
    scalar_value = float(raw) if isinstance(raw, (int, float)) else 1.0
    raw_input_dtype = node_input_dtype(node, 0)
    input_dtype = "int64" if raw_input_dtype == "int64" else activation_dtype
    contract_shape = shape_to_contract(out_shape)
    n_expr = flat_numel_expr(contract_shape)
    return ShaderVariant(
        name="div_scalar",
        family="export",
        contract=ShaderContract(
            class_name="ExportDivScalarProgram",
            shader_name="div_scalar",
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
        execution_requirements=_scalar_requirements(input_dtype, activation_dtype),
    )


def _scalar_source(input_dtype: str, activation_dtype: str) -> str:
    return (
        _SCALAR_SOURCE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{INPUT_EXTENSION}}", _input_extension_source(input_dtype, activation_dtype))
        .replace("{{INPUT_TYPE}}", _input_glsl_type(input_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_DIV}}", activation_store("float(x[idx]) / pc.scalar", activation_dtype))
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


def _scalar_requirements(input_dtype: str, output_dtype: str) -> ShaderExecutionRequirements | None:
    requirements = activation_requirements(output_dtype)
    if input_dtype != "int64":
        return requirements
    if requirements is None:
        return ShaderExecutionRequirements(require_shader_int64=True)
    return ShaderExecutionRequirements(
        subgroup=requirements.subgroup,
        cooperative_matrix=requirements.cooperative_matrix,
        require_integer_dot_product=requirements.require_integer_dot_product,
        require_shader_int64=True,
        require_buffer_device_address=requirements.require_buffer_device_address,
        require_storage_buffer_16bit_access=requirements.require_storage_buffer_16bit_access,
    )
