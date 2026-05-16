from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    activation_store,
    flat_numel_expr,
    node_output_shape,
    shape_to_contract,
)
from torch2vk.runtime.shader import (
    ExprDim,
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
from torch2vk.vulkan.types import Dim
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements

_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; float start; float step; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = {{STORE_ARANGE}}; }
}
"""

_INT64_SOURCE = """\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { int64_t output_values[]; };
layout(push_constant) uniform PushConstants { uint N; int start; int step; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        output_values[idx] = int64_t(pc.start) + int64_t(idx) * int64_t(pc.step);
    }
}
"""


def make_arange_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None
    start, step = _arange_start_step(node)
    output_dtype = _node_output_dtype(node) or activation_dtype
    contract_shape = shape_to_contract(out_shape)
    n_expr = flat_numel_expr(contract_shape)
    if output_dtype == "int64":
        return _make_arange_int64(contract_shape, n_expr, start, step)
    return ShaderVariant(
        name="arange_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportArangeProgram",
            shader_name="arange_f32",
            fields=(
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=contract_shape),
                ),
            ),
            push_constants=PushConstantSpec(
                size=12,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),
                    PushConstantFieldSpec("start", PushConstantType.FLOAT32, 4, start),
                    PushConstantFieldSpec("step", PushConstantType.FLOAT32, 8, step),
                ),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_source(activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _node_output_dtype(node: Node) -> str:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
        return ""
    return str(tensor_meta.dtype).removeprefix("torch.")


def _make_arange_int64(
    contract_shape: tuple[Dim, ...],
    n_expr: ExprDim,
    start: float,
    step: float,
) -> ShaderVariant:
    return ShaderVariant(
        name="arange_i64",
        family="export",
        contract=ShaderContract(
            class_name="ExportArangeI64Program",
            shader_name="arange_i64",
            fields=(
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype="int64", shape=contract_shape),
                ),
            ),
            push_constants=PushConstantSpec(
                size=12,
                fields=(
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n_expr),
                    PushConstantFieldSpec("start", PushConstantType.INT32, 4, int(start)),
                    PushConstantFieldSpec("step", PushConstantType.INT32, 8, int(step)),
                ),
            ),
            dispatch=(ceil_div(n_expr, 256), 1, 1),
        ),
        source=_INT64_SOURCE,
        execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    )


def _arange_start_step(node: Node) -> tuple[float, float]:
    if str(node.target) == "aten.arange.start_step":
        start = node.args[0] if len(node.args) > 0 and isinstance(node.args[0], (int, float)) else 0
        step = node.args[2] if len(node.args) > 2 and isinstance(node.args[2], (int, float)) else 1
        return float(start), float(step)
    if str(node.target) == "aten.arange.start":
        start = node.args[0] if len(node.args) > 0 and isinstance(node.args[0], (int, float)) else 0
        return float(start), 1.0
    return 0.0, 1.0


def _source(activation_dtype: str) -> str:
    return (
        _SOURCE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_ARANGE}}", activation_store("pc.start + float(idx) * pc.step", activation_dtype))
    )
