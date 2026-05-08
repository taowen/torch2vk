"""Generated shader: decode_embed_export_embedding_f32."""

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
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


DECODE_EMBED_EXPORT_EMBEDDING_F32 = ShaderVariant(
    name='decode_embed_export_embedding_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportEmbeddingProgram',
        shader_name='decode_embed_export_embedding_f32',
        fields=(
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='bfloat16', shape=('W0', 'W1',)),
            ),
            TensorFieldSpec(
                name='indices',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='int64', shape=('I0', 'I1',)),
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
                PushConstantFieldSpec('num_indices', PushConstantType.UINT32, 0, 1, dynamic=False),
                PushConstantFieldSpec('embedding_dim', PushConstantType.UINT32, 4, 1024, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(1024, 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""\
#version 450
#extension GL_EXT_bfloat16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly WeightBuffer { bfloat16_t weight[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndicesBuffer { int64_t indices[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint num_indices; uint embedding_dim; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.num_indices * pc.embedding_dim) return;
    const uint token_idx = idx / pc.embedding_dim;
    const uint dim_idx = idx - token_idx * pc.embedding_dim;
    const int64_t token_id = indices[token_idx];
    output_values[idx] = float(weight[uint(token_id) * pc.embedding_dim + dim_idx]);
}
""",
)
