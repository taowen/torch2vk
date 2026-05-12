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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements

_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
{{WEIGHT_EXTENSION}}\
{{INDEX_EXTENSION}}
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly WeightBuffer { {{WEIGHT_TYPE}} weight[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndicesBuffer { {{INDEX_TYPE}} indices[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint num_indices; uint embedding_dim; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.num_indices * pc.embedding_dim) return;
    const uint token_idx = idx / pc.embedding_dim;
    const uint dim_idx = idx - token_idx * pc.embedding_dim;
    const {{INDEX_TYPE}} token_id = indices[token_idx];
    output_values[idx] = {{STORE_EMBEDDING}};
}
"""


def make_embedding_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    out_shape = node_output_shape(node)
    if not out_shape:
        return None

    weight_shape = node_input_shape(node, 0)
    indices_shape = node_input_shape(node, 1)
    if not weight_shape or not indices_shape:
        return None
    indices_dtype = node_input_dtype(node, 1)
    if indices_dtype not in {"int32", "int64"}:
        return None

    weight_dtype = node_input_dtype(node, 0)
    weight_suffix = weight_dtype_suffix(weight_dtype)
    shader_name = f"embedding_{weight_suffix}w_f32"
    embedding_dim = weight_shape[-1]
    num_indices = 1
    for d in indices_shape:
        num_indices *= d

    w_contract = tuple(f"W{i}" for i in range(len(weight_shape)))
    idx_contract = tuple(f"I{i}" for i in range(len(indices_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))

    total = num_indices * embedding_dim
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name=f"ExportEmbedding{weight_suffix.title()}WeightProgram",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec(
                    "weight",
                    IOKind.INPUT,
                    "weight",
                    TensorContract(dtype=weight_dtype, shape=w_contract),
                ),
                TensorFieldSpec(
                    "indices",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=indices_dtype, shape=idx_contract),
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
                    PushConstantFieldSpec("num_indices", PushConstantType.UINT32, 0, num_indices),
                    PushConstantFieldSpec(
                        "embedding_dim", PushConstantType.UINT32, 4, embedding_dim
                    ),
                ),
            ),
            dispatch=(ceil_div(total, 256), 1, 1),
        ),
        execution_requirements=activation_requirements(
            activation_dtype,
            _index_execution_requirements(indices_dtype),
        ),
        source=_index_source(
            indices_dtype, weight_dtype=weight_dtype, activation_dtype=activation_dtype
        ),
    )


def _index_source(dtype: str, *, weight_dtype: str, activation_dtype: str) -> str:
    index_type = "int64_t" if dtype == "int64" else "int"
    extension = (
        "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require"
        if dtype == "int64"
        else ""
    )
    weight_value = "float(weight[uint(token_id) * pc.embedding_dim + dim_idx])"
    return (
        _SOURCE.replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{WEIGHT_EXTENSION}}", weight_extension_source(weight_dtype))
        .replace("{{WEIGHT_TYPE}}", weight_glsl_type(weight_dtype))
        .replace("{{INDEX_EXTENSION}}", extension)
        .replace("{{INDEX_TYPE}}", index_type)
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_EMBEDDING}}", activation_store(weight_value, activation_dtype))
    )


def _index_execution_requirements(dtype: str) -> ShaderExecutionRequirements | None:
    if dtype == "int64":
        return ShaderExecutionRequirements(require_shader_int64=True)
    return None
