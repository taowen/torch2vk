from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    activation_store,
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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements

_SOURCE = """\
#version 450
{{EXTENSIONS}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{X_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { {{Y_TYPE}} y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{OUT_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N_OUT; uint D; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N_OUT) {
        const uint d = idx % pc.D;
        const uint x_idx = idx / pc.D;
        output_values[idx] = {{STORE_OUT}};
    }
}
"""


def make_einsum_outer_variant(
    node: Node,
    activation_dtype: str = "float32",
) -> ShaderVariant | None:
    if not node.args or node.args[0] != "...n,d->...nd":
        return None
    inputs = node.args[1] if len(node.args) > 1 else None
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
        return None
    x_node, y_node = inputs
    if not isinstance(x_node, Node) or not isinstance(y_node, Node):
        return None
    x_meta = x_node.meta.get("tensor_meta")
    y_meta = y_node.meta.get("tensor_meta")
    out_shape = node_output_shape(node)
    if x_meta is None or y_meta is None or not out_shape:
        return None
    x_shape = tuple(int(d) for d in x_meta.shape)
    y_shape = tuple(int(d) for d in y_meta.shape)
    if len(y_shape) != 1:
        return None
    x_dtype = str(x_meta.dtype).removeprefix("torch.")
    y_dtype = str(y_meta.dtype).removeprefix("torch.")
    if y_dtype == "float32":
        y_dtype = activation_dtype
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))
    x_contract = tuple(f"X{i}" for i in range(len(x_shape)))
    y_contract = tuple(f"Y{i}" for i in range(len(y_shape)))
    n_out = product_expr(out_contract)
    return ShaderVariant(
        name="einsum_outer_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportEinsumOuterProgram",
            shader_name="einsum_outer_f32",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype=x_dtype, shape=x_contract)),
                TensorFieldSpec("y", IOKind.INPUT, "input", TensorContract(dtype=y_dtype, shape=y_contract)),
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
                    PushConstantFieldSpec("N_OUT", PushConstantType.UINT32, 0, n_out),
                    PushConstantFieldSpec("D", PushConstantType.UINT32, 4, out_contract[-1]),
                ),
            ),
            dispatch=(ceil_div(n_out, 256), 1, 1),
        ),
        source=_source(x_dtype=x_dtype, y_dtype=y_dtype, output_dtype=activation_dtype),
        execution_requirements=_requirements(x_dtype, y_dtype, activation_dtype),
    )


def _source(*, x_dtype: str, y_dtype: str, output_dtype: str) -> str:
    return (
        _SOURCE.replace("{{EXTENSIONS}}", _extensions(x_dtype, y_dtype, output_dtype))
        .replace("{{X_TYPE}}", _glsl_type(x_dtype))
        .replace("{{Y_TYPE}}", _glsl_type(y_dtype))
        .replace("{{OUT_TYPE}}", activation_glsl_type(output_dtype))
        .replace(
            "{{STORE_OUT}}",
            activation_store("float(x[x_idx]) * float(y[d])", output_dtype),
        )
    )


def _glsl_type(dtype: str) -> str:
    if dtype == "float16":
        return "float16_t"
    if dtype == "float32":
        return "float"
    if dtype == "int64":
        return "int64_t"
    if dtype == "int32":
        return "int"
    raise ValueError(f"Unsupported einsum input dtype: {dtype}")


def _extensions(*dtypes: str) -> str:
    extensions: list[str] = []
    if "float16" in dtypes:
        extensions.append(activation_extension_source("float16").rstrip())
    if "int64" in dtypes:
        extensions.append("#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require")
    return "\n".join(extensions) + ("\n" if extensions else "")


def _requirements(*dtypes: str) -> ShaderExecutionRequirements | None:
    requirements = activation_requirements("float16") if "float16" in dtypes else None
    if "int64" not in dtypes:
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
