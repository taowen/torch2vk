"""Generated shader: embedding_q8_0_f32."""

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
    mul,
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)
from torch2vk.vulkan.types import (
    q8_0_halfwords_layout,
)


EMBEDDING_Q8_0_F32 = ShaderVariant(
    name='embedding_q8_0_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportEmbeddingQ8_0Program',
        shader_name='embedding_q8_0_f32',
        fields=(
            TensorFieldSpec(
                name='weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='uint16', shape=('V', 544,), layout=q8_0_halfwords_layout(logical_k='H', block_size=32, halfwords_per_block=17)),
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
                contract=TensorContract(dtype='float32', shape=('I0', 'I1', 'H',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('num_indices', PushConstantType.UINT32, 0, mul('I0', 'I1'), dynamic=False),
                PushConstantFieldSpec('embedding_dim', PushConstantType.UINT32, 4, 'H', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('I0', 'I1'), 'H'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True, require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndicesBuffer { int64_t indices[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint num_indices; uint embedding_dim; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

float q8_0_value(uint row, uint h) {
    const uint blocks_per_row = pc.embedding_dim / 32u;
    const uint block_index = h >> 5u;
    const uint block_half = row * blocks_per_row * 17u + block_index * 17u;
    const float d = unpackHalf2x16(uint(weight[block_half])).x;
    const uint local = h & 31u;
    const uint packed = uint(weight[block_half + 1u + (local >> 1u)]);
    uint byte_value = ((local & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
    int quant = int(byte_value);
    if (quant >= 128) {
        quant -= 256;
    }
    return d * float(quant);
}

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.num_indices * pc.embedding_dim) return;
    const uint token_idx = idx / pc.embedding_dim;
    const uint dim_idx = idx - token_idx * pc.embedding_dim;
    const int64_t token_id = indices[token_idx];
    output_values[idx] = q8_0_value(uint(token_id), dim_idx);
}
""",
)
