from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    node_input_shape,
    node_output_shape,
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

_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { int output_values[]; };
layout(push_constant) uniform PushConstants { uint num_rows; uint row_len; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint row = gl_GlobalInvocationID.x;
    if (row >= pc.num_rows) return;
    const uint base = row * pc.row_len;
    float max_val = float(x[base]);
    int max_idx = 0;
    for (uint i = 1u; i < pc.row_len; ++i) {
        const float val = float(x[base + i]);
        if (val > max_val) { max_val = val; max_idx = int(i); }
    }
    output_values[row] = max_idx;
}
"""


def make_argmax_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    x_shape = node_input_shape(node, 0)
    out_shape = node_output_shape(node)
    if not x_shape:
        return None

    row_len = x_shape[-1]
    num_rows = 1
    for d in x_shape[:-1]:
        num_rows *= d

    x_contract = tuple(f"X{i}" for i in range(len(x_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape))) if out_shape else ("N",)

    return ShaderVariant(
        name="argmax_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportArgmaxProgram",
            shader_name="argmax_f32",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype=activation_dtype, shape=x_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="int32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=8,
                fields=(
                    PushConstantFieldSpec("num_rows", PushConstantType.UINT32, 0, num_rows),
                    PushConstantFieldSpec("row_len", PushConstantType.UINT32, 4, row_len),
                ),
            ),
            dispatch=(ceil_div(num_rows, 256), 1, 1),
        ),
        source=_source(activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _source(activation_dtype: str) -> str:
    return (
        _SOURCE
        .replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
    )
