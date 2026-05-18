from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    activation_store,
    node_input_dtype,
    node_input_shape,
    node_output_shape,
    product_expr,
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
)

_SOURCE_TEMPLATE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
{{WEIGHT_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
{{WEIGHT_BUFFER}}\
layout(set = 0, binding = {{OUTPUT_BINDING}}) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint ROWS; uint COLS; float eps; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float partial_sumsq[256];
void main() {
    const uint row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    if (row >= pc.ROWS) { return; }
    float sumsq = 0.0;
    for (uint c = tid; c < pc.COLS; c += 256u) {
        float v = float(x[row * pc.COLS + c]);
        sumsq += v * v;
    }
    partial_sumsq[tid] = sumsq;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        barrier();
    }
    float inv_rms = inversesqrt(partial_sumsq[0] / float(pc.COLS) + pc.eps);
    for (uint c = tid; c < pc.COLS; c += 256u) {
        uint idx = row * pc.COLS + c;
        output_values[idx] = {{STORE_NORM}};
    }
}
"""


def make_rms_norm_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    in_shape = node_input_shape(node, 0)
    out_shape = node_output_shape(node)
    if not in_shape or not out_shape or in_shape != out_shape:
        return None

    normalized_shape = _normalized_shape(node.args[1] if len(node.args) > 1 else None)
    if len(normalized_shape) != 1 or normalized_shape[0] != in_shape[-1]:
        return None

    eps_val = 1e-5
    if len(node.args) > 3 and isinstance(node.args[3], (int, float)):
        eps_val = float(node.args[3])
    else:
        eps_arg = node.kwargs.get("eps")
        if isinstance(eps_arg, (int, float)):
            eps_val = float(eps_arg)

    in_contract = tuple(f"I{i}" for i in range(len(in_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))
    rows = product_expr(in_contract[:-1])
    cols_expr = in_contract[-1]
    has_weight = len(node.args) > 2 and isinstance(node.args[2], Node)
    weight_dtype = node_input_dtype(node, 2) if has_weight else ""
    weight_suffix = weight_dtype_suffix(weight_dtype) if has_weight else "none"
    shader_name = f"rms_norm_{weight_suffix}w_f32"
    fields = [
        TensorFieldSpec(
            "x",
            IOKind.INPUT,
            "input",
            TensorContract(dtype=activation_dtype, shape=in_contract),
        )
    ]
    if has_weight:
        fields.append(
            TensorFieldSpec(
                "weight",
                IOKind.INPUT,
                "weight",
                TensorContract(dtype=weight_dtype, shape=("C",)),
            )
        )
    fields.append(
        TensorFieldSpec(
            "output",
            IOKind.OUTPUT,
            "output",
            TensorContract(dtype=activation_dtype, shape=out_contract),
        )
    )

    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"ExportRmsNorm{weight_suffix.title()}WeightProgram",
            shader_name=shader_name,
            fields=tuple(fields),
            push_constants=PushConstantSpec(
                size=12,
                fields=(
                    PushConstantFieldSpec("ROWS", PushConstantType.UINT32, 0, rows),
                    PushConstantFieldSpec("COLS", PushConstantType.UINT32, 4, cols_expr),
                    PushConstantFieldSpec("eps", PushConstantType.FLOAT32, 8, eps_val),
                ),
            ),
            dispatch=(rows, 1, 1),
        ),
        source=_source(
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            has_weight=has_weight,
        ),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _normalized_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value):
        return tuple(value)
    return ()


def _source(*, weight_dtype: str, activation_dtype: str, has_weight: bool) -> str:
    weight_buffer = ""
    weight_extension = ""
    output_binding = "1"
    norm_expr = "float(x[idx]) * inv_rms"
    if has_weight:
        weight_extension = weight_extension_source(weight_dtype)
        weight_buffer = (
            "layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer "
            f"{{ {weight_glsl_type(weight_dtype)} weight[]; }};\n"
        )
        output_binding = "2"
        norm_expr = f"{norm_expr} * float(weight[c])"
    return (
        _SOURCE_TEMPLATE.replace(
            "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
        )
        .replace("{{WEIGHT_EXTENSION}}", weight_extension)
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{WEIGHT_BUFFER}}", weight_buffer)
        .replace("{{OUTPUT_BINDING}}", output_binding)
        .replace("{{STORE_NORM}}", activation_store(norm_expr, activation_dtype))
    )
