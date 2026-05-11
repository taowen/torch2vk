"""Generated shader: qwen3_token_store_eos_f32."""

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
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


QWEN3_TOKEN_STORE_EOS_F32 = ShaderVariant(
    name='qwen3_token_store_eos_f32',
    family='qwen3.text',
    contract=ShaderContract(
        class_name='Qwen3TokenStoreEosF32Program',
        shader_name='qwen3_token_store_eos_f32',
        fields=(
            TensorFieldSpec(
                name='next_token',
                io_kind=IOKind.INPUT,
                role='next_token',
                contract=TensorContract(dtype='int64', shape=(1, 1,)),
            ),
            TensorFieldSpec(
                name='token_index',
                io_kind=IOKind.INPUT,
                role='token_index',
                contract=TensorContract(dtype='int64', shape=(1,)),
            ),
            TensorFieldSpec(
                name='done',
                io_kind=IOKind.INPUT,
                role='done',
                contract=TensorContract(dtype='uint32', shape=(1,)),
            ),
            TensorFieldSpec(
                name='generated_tokens',
                io_kind=IOKind.INOUT,
                role='generated_tokens',
                contract=TensorContract(dtype='int64', shape=(1, 'G',)),
            ),
            TensorFieldSpec(
                name='generated_length',
                io_kind=IOKind.INOUT,
                role='generated_length',
                contract=TensorContract(dtype='uint32', shape=(1,)),
            ),
            TensorFieldSpec(
                name='stopped',
                io_kind=IOKind.INOUT,
                role='stopped',
                contract=TensorContract(dtype='uint32', shape=(1,)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('G', PushConstantType.UINT32, 0, 'G', dynamic=False),
                PushConstantFieldSpec('stop_on_eos', PushConstantType.UINT32, 4, 1, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(1, 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly NextTokenBuffer {
    int64_t next_token[];
};

layout(set = 0, binding = 1) buffer restrict readonly TokenIndexBuffer {
    int64_t token_index[];
};

layout(set = 0, binding = 2) buffer restrict readonly DoneBuffer {
    uint done[];
};

layout(set = 0, binding = 3) buffer restrict GeneratedTokensBuffer {
    int64_t generated_tokens[];
};

layout(set = 0, binding = 4) buffer restrict GeneratedLengthBuffer {
    uint generated_length[];
};

layout(set = 0, binding = 5) buffer restrict StoppedBuffer {
    uint stopped[];
};

layout(push_constant) uniform PushConstants {
    uint G;
    uint stop_on_eos;
} pc;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint index = uint(token_index[0]);
    if (index >= pc.G) {
        return;
    }
    if (pc.stop_on_eos != 0u && stopped[0] != 0u) {
        return;
    }

    generated_tokens[index] = next_token[0];
    generated_length[0] = index + 1u;
    if (pc.stop_on_eos != 0u && done[0] != 0u) {
        stopped[0] = 1u;
    }
}
""",
)
