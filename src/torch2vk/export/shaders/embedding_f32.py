from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import node_input_shape, node_output_shape
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
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndicesBuffer { int indices[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint num_indices; uint embedding_dim; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.num_indices * pc.embedding_dim) return;
    const uint token_idx = idx / pc.embedding_dim;
    const uint dim_idx = idx - token_idx * pc.embedding_dim;
    const int token_id = indices[token_idx];
    output_values[idx] = weight[uint(token_id) * pc.embedding_dim + dim_idx];
}
"""


def make_embedding_variant(node: Node) -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None

    weight_shape = node_input_shape(node, 0)
    indices_shape = node_input_shape(node, 1)
    if not weight_shape or not indices_shape:
        return None

    embedding_dim = weight_shape[-1]
    num_indices = 1
    for d in indices_shape:
        num_indices *= d

    w_contract = tuple(f"W{i}" for i in range(len(weight_shape)))
    idx_contract = tuple(f"I{i}" for i in range(len(indices_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))

    total = num_indices * embedding_dim
    return ShaderVariant(
        name="export_embedding_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportEmbeddingProgram",
            shader_name="export_embedding_f32",
            fields=(
                TensorFieldSpec("weight", IOKind.INPUT, "weight", TensorContract(dtype="float32", shape=w_contract)),
                TensorFieldSpec("indices", IOKind.INPUT, "input", TensorContract(dtype="int32", shape=idx_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=8,
                fields=(
                    PushConstantFieldSpec("num_indices", PushConstantType.UINT32, 0, num_indices),
                    PushConstantFieldSpec("embedding_dim", PushConstantType.UINT32, 4, embedding_dim),
                ),
            ),
            dispatch=(ceil_div(total, 256), 1, 1),
        ),
        source=_SOURCE,
    )
