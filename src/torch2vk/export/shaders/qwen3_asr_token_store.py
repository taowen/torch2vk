"""Qwen3-ASR generated-token store shader for replay decode."""

from __future__ import annotations

from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantInput,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderExecutionRequirements,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
)


QWEN3_ASR_TOKEN_STORE = ShaderVariant(
    name="qwen3_asr_token_store",
    family="qwen3_asr.text",
    contract=ShaderContract(
        class_name="Qwen3AsrTokenStoreProgram",
        shader_name="qwen3_asr_token_store",
        fields=(
            TensorFieldSpec(
                name="next_token",
                io_kind=IOKind.INPUT,
                role="next_token",
                contract=TensorContract(dtype="int64", shape=(1, 1)),
            ),
            TensorFieldSpec(
                name="done",
                io_kind=IOKind.INPUT,
                role="done",
                contract=TensorContract(dtype="uint32", shape=(1,)),
            ),
            TensorFieldSpec(
                name="generated_tokens",
                io_kind=IOKind.INOUT,
                role="generated_tokens",
                contract=TensorContract(dtype="int64", shape=(1, "G")),
            ),
            TensorFieldSpec(
                name="generated_length",
                io_kind=IOKind.INOUT,
                role="generated_length",
                contract=TensorContract(dtype="uint32", shape=(1,)),
            ),
            TensorFieldSpec(
                name="stopped",
                io_kind=IOKind.INOUT,
                role="stopped",
                contract=TensorContract(dtype="uint32", shape=(1,)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("G", PushConstantType.UINT32, 0, "G"),
                PushConstantFieldSpec("stop_on_eos", PushConstantType.UINT32, 4, 0),
                PushConstantFieldSpec(
                    "token_index", PushConstantType.UINT32, 8, PushConstantInput("token_index")
                ),
            ),
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

layout(set = 0, binding = 1) buffer restrict readonly DoneBuffer {
    uint done[];
};

layout(set = 0, binding = 2) buffer restrict GeneratedTokensBuffer {
    int64_t generated_tokens[];
};

layout(set = 0, binding = 3) buffer restrict GeneratedLengthBuffer {
    uint generated_length[];
};

layout(set = 0, binding = 4) buffer restrict StoppedBuffer {
    uint stopped[];
};

layout(push_constant) uniform PushConstants {
    uint G;
    uint stop_on_eos;
    uint token_index;
} pc;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint index = pc.token_index;
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
""".lstrip(),
)


QWEN3_ASR_TOKEN_STORE_EOS = ShaderVariant(
    name="qwen3_asr_token_store_eos",
    family=QWEN3_ASR_TOKEN_STORE.family,
    contract=ShaderContract(
        class_name="Qwen3AsrTokenStoreEosProgram",
        shader_name="qwen3_asr_token_store_eos",
        fields=QWEN3_ASR_TOKEN_STORE.contract.fields,
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("G", PushConstantType.UINT32, 0, "G"),
                PushConstantFieldSpec("stop_on_eos", PushConstantType.UINT32, 4, 1),
                PushConstantFieldSpec(
                    "token_index", PushConstantType.UINT32, 8, PushConstantInput("token_index")
                ),
            ),
        ),
        dispatch=QWEN3_ASR_TOKEN_STORE.contract.dispatch,
    ),
    execution_requirements=QWEN3_ASR_TOKEN_STORE.execution_requirements,
    source=QWEN3_ASR_TOKEN_STORE.source,
)
