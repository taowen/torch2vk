from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
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
layout(set = 0, binding = 0) buffer restrict readonly ABuffer { {{ACTIVATION_TYPE}} a[]; };
layout(set = 0, binding = 1) buffer restrict readonly BBuffer { {{ACTIVATION_TYPE}} b[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N_OUT; uint A_STRIDE; uint B_STRIDE; uint OUT_STRIDE; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N_OUT) {
        uint row = idx / pc.OUT_STRIDE;
        uint col = idx % pc.OUT_STRIDE;
        if (col < pc.A_STRIDE) {
            output_values[idx] = a[row * pc.A_STRIDE + col];
        } else {
            output_values[idx] = b[row * pc.B_STRIDE + (col - pc.A_STRIDE)];
        }
    }
}
"""


def make_cat_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None

    inputs = node.args[0]
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
        return None

    a_node, b_node = inputs[0], inputs[1]
    if not isinstance(a_node, Node) or not isinstance(b_node, Node):
        return None

    a_meta = a_node.meta.get("tensor_meta")
    b_meta = b_node.meta.get("tensor_meta")
    if a_meta is None or b_meta is None:
        return None

    a_shape = tuple(int(d) for d in a_meta.shape)
    b_shape = tuple(int(d) for d in b_meta.shape)

    dim = int(node.args[1]) if len(node.args) > 1 and isinstance(node.args[1], int) else 0
    if dim < 0:
        dim = len(out_shape) + dim

    a_dim_stride = 1
    for d in a_shape[dim + 1 :]:
        a_dim_stride *= d
    b_dim_stride = 1
    for d in b_shape[dim + 1 :]:
        b_dim_stride *= d

    a_stride = a_shape[dim] * a_dim_stride
    b_stride = b_shape[dim] * b_dim_stride
    out_stride = out_shape[dim] * a_dim_stride

    a_contract = tuple(f"A{i}" for i in range(len(a_shape)))
    b_contract = tuple(f"B{i}" for i in range(len(b_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))
    n_out = product_expr(out_contract)

    return ShaderVariant(
        name="cat_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportCatProgram",
            shader_name="cat_f32",
            fields=(
                TensorFieldSpec(
                    "a",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=a_contract),
                ),
                TensorFieldSpec(
                    "b",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=b_contract),
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=out_contract),
                ),
            ),
            push_constants=PushConstantSpec(
                size=16,
                fields=(
                    PushConstantFieldSpec("N_OUT", PushConstantType.UINT32, 0, n_out),
                    PushConstantFieldSpec("A_STRIDE", PushConstantType.UINT32, 4, a_stride),
                    PushConstantFieldSpec("B_STRIDE", PushConstantType.UINT32, 8, b_stride),
                    PushConstantFieldSpec("OUT_STRIDE", PushConstantType.UINT32, 12, out_stride),
                ),
            ),
            dispatch=(ceil_div(n_out, 256), 1, 1),
        ),
        source=_source(activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _source(activation_dtype: str) -> str:
    return _SOURCE.replace(
        "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
    ).replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
