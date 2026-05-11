from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
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

_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N_OUT; uint IN_STRIDE; uint OUT_STRIDE; uint OFFSET; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N_OUT) {
        uint row = idx / pc.OUT_STRIDE;
        uint col = idx % pc.OUT_STRIDE;
        output_values[idx] = x[row * pc.IN_STRIDE + pc.OFFSET + col];
    }
}
"""


def make_slice_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    in_shape = node_input_shape(node, 0)
    out_shape = node_output_shape(node)
    if not in_shape or not out_shape:
        return None

    dim = int(node.args[1]) if len(node.args) > 1 and isinstance(node.args[1], int) else 0
    start = int(node.args[2]) if len(node.args) > 2 and isinstance(node.args[2], int) else 0

    if dim < 0:
        dim = len(in_shape) + dim

    in_dim_size = in_shape[dim]
    out_dim_size = out_shape[dim]

    in_stride = 1
    for d in in_shape[dim + 1:]:
        in_stride *= d
    out_stride = 1
    for d in out_shape[dim + 1:]:
        out_stride *= d

    if dim == len(in_shape) - 1:
        in_stride_val = in_dim_size
        out_stride_val = out_dim_size
        offset = start
    else:
        in_stride_val = in_dim_size * in_stride
        out_stride_val = out_dim_size * out_stride
        offset = start * in_stride

    in_contract = tuple(f"I{i}" for i in range(len(in_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))
    n_out = product_expr(out_contract)

    return ShaderVariant(
        name="slice_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportSliceProgram",
            shader_name="slice_f32",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype=activation_dtype, shape=in_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype=activation_dtype, shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=16,
                fields=(
                    PushConstantFieldSpec("N_OUT", PushConstantType.UINT32, 0, n_out),
                    PushConstantFieldSpec("IN_STRIDE", PushConstantType.UINT32, 4, in_stride_val),
                    PushConstantFieldSpec("OUT_STRIDE", PushConstantType.UINT32, 8, out_stride_val),
                    PushConstantFieldSpec("OFFSET", PushConstantType.UINT32, 12, offset),
                ),
            ),
            dispatch=(ceil_div(n_out, 256), 1, 1),
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
