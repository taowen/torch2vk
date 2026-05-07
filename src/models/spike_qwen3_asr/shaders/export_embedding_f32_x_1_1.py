"""Generated shader: export_embedding_f32_x_1_1."""

from __future__ import annotations

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


EXPORT_EMBEDDING_F32_X_1_1 = ShaderVariant(
    name='export_embedding_f32_x_1_1',
    family='export',
    contract=ShaderContract(
        class_name='ExportEmbeddingProgram',
        shader_name='export_embedding_f32_x_1_1',
        fields=(
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='float32', shape=('W0', 'W1',)),
            ),
            TensorFieldSpec(
                name='indices',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='int32', shape=('I0', 'I1',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O0', 'O1', 'O2',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('num_indices', PushConstantType.UINT32, 0, 1),
                PushConstantFieldSpec('embedding_dim', PushConstantType.UINT32, 4, 1024),
            ),
        ),
        dispatch=(ceil_div(1024, 256), 1, 1),
    ),
    source="""\
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
""",
)
