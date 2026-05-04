"""Qwen3-ASR generated-token store shader for replay decode."""

from __future__ import annotations

from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderExecutionRequirements,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
)


QWEN3_ASR_TOKEN_STORE_F32 = ShaderVariant(
    name="qwen3_asr_token_store_f32",
    family="qwen3_asr.text",
    contract=ShaderContract(
        class_name="Qwen3AsrTokenStoreF32Program",
        shader_name="qwen3_asr_token_store_f32",
        fields=(
            TensorFieldSpec(
                name="next_token",
                io_kind=IOKind.INPUT,
                role="next_token",
                contract=TensorContract(dtype="int64", shape=(1,)),
            ),
            TensorFieldSpec(
                name="token_index",
                io_kind=IOKind.INPUT,
                role="token_index",
                contract=TensorContract(dtype="int64", shape=(1,)),
            ),
            TensorFieldSpec(
                name="generated_tokens",
                io_kind=IOKind.INOUT,
                role="generated_tokens",
                contract=TensorContract(dtype="int64", shape=(1, "G")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(PushConstantFieldSpec("G", PushConstantType.UINT32, 0, "G"),),
        ),
        dispatch=(1, 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly NextTokenBuffer {
    int64_t next_token[];
};

layout(set = 0, binding = 1) buffer restrict readonly TokenIndexBuffer {
    int64_t token_index[];
};

layout(set = 0, binding = 2) buffer restrict GeneratedTokensBuffer {
    int64_t generated_tokens[];
};

layout(push_constant) uniform PushConstants {
    uint G;
} pc;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint index = uint(token_index[0]);
    if (index < pc.G) {
        generated_tokens[index] = next_token[0];
    }
}
""".lstrip(),
)
