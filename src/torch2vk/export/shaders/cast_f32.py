from __future__ import annotations

import hashlib

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_io_requirements,
    node_input_dtype,
    node_input_shape,
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


_SOURCE_TEMPLATE = """\
#version 450
{{EXTENSIONS}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{INPUT_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{OUTPUT_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    output_values[idx] = {{OUTPUT_TYPE}}(x[idx]);
}
"""


def make_cast_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    input_shape = node_input_shape(node, 0)
    output_shape = node_output_shape(node)
    if not input_shape or input_shape != output_shape:
        return None
    input_dtype = node_input_dtype(node, 0)
    output_dtype = _node_dtype(node)
    if input_dtype == output_dtype:
        return None
    if input_dtype not in {"float16", "float32"} or output_dtype not in {"float16", "float32"}:
        return None

    rank = len(output_shape)
    contract_shape = tuple(f"D{i}" for i in range(rank))
    n_expr = product_expr(contract_shape)
    shader_name = _shader_name(input_shape, input_dtype, output_dtype)
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name="ExportCastProgram",
            shader_name=shader_name,
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
                size=4,
                fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_source(input_dtype, output_dtype),
        execution_requirements=activation_io_requirements(input_dtype, output_dtype),
    )


def _shader_name(shape: tuple[int, ...], input_dtype: str, output_dtype: str) -> str:
    payload = f"{shape}:{input_dtype}->{output_dtype}".encode()
    digest = hashlib.sha1(payload).hexdigest()[:10]
    return f"cast_{_dtype_suffix(input_dtype)}_to_{_dtype_suffix(output_dtype)}_{digest}"


def _source(input_dtype: str, output_dtype: str) -> str:
    extensions = "".join(
        dict.fromkeys(
            (
                activation_extension_source(input_dtype),
                activation_extension_source(output_dtype),
            )
        )
    )
    return (
        _SOURCE_TEMPLATE.replace("{{EXTENSIONS}}", extensions)
        .replace("{{INPUT_TYPE}}", activation_glsl_type(input_dtype))
        .replace("{{OUTPUT_TYPE}}", activation_glsl_type(output_dtype))
    )


def _dtype_suffix(dtype: str) -> str:
    if dtype == "float16":
        return "f16"
    if dtype == "float32":
        return "f32"
    raise ValueError(f"Unsupported cast dtype: {dtype}")


def _node_dtype(node: Node) -> str:
    tm = node.meta.get("tensor_meta")
    if tm is None:
        return ""
    return str(tm.dtype).removeprefix("torch.")
