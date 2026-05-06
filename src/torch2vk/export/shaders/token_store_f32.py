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

TOKEN_STORE_F32 = ShaderVariant(
    name="token_store_f32",
    family="text",
    contract=ShaderContract(
        class_name="TokenStoreF32Program",
        shader_name="token_store_f32",
        fields=(
            TensorFieldSpec("next_token", IOKind.INPUT, "next_token", TensorContract(dtype="int64", shape=(1,))),
            TensorFieldSpec("token_index", IOKind.INPUT, "token_index", TensorContract(dtype="int64", shape=(1,))),
            TensorFieldSpec("done", IOKind.INPUT, "done", TensorContract(dtype="uint32", shape=(1,))),
            TensorFieldSpec("generated_tokens", IOKind.INOUT, "generated_tokens", TensorContract(dtype="int64", shape=(1, "G"))),
            TensorFieldSpec("generated_length", IOKind.INOUT, "generated_length", TensorContract(dtype="uint32", shape=(1,))),
            TensorFieldSpec("stopped", IOKind.INOUT, "stopped", TensorContract(dtype="uint32", shape=(1,))),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("G", PushConstantType.UINT32, 0, "G"),
                PushConstantFieldSpec("stop_on_eos", PushConstantType.UINT32, 4, 0),
            ),
        ),
        dispatch=(1, 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly NextTokenBuffer { int64_t next_token[]; };
layout(set = 0, binding = 1) buffer restrict readonly TokenIndexBuffer { int64_t token_index[]; };
layout(set = 0, binding = 2) buffer restrict readonly DoneBuffer { uint done[]; };
layout(set = 0, binding = 3) buffer restrict GeneratedTokensBuffer { int64_t generated_tokens[]; };
layout(set = 0, binding = 4) buffer restrict GeneratedLengthBuffer { uint generated_length[]; };
layout(set = 0, binding = 5) buffer restrict StoppedBuffer { uint stopped[]; };
layout(push_constant) uniform PushConstants { uint G; uint stop_on_eos; } pc;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  const uint index = uint(token_index[0]);
  if (index >= pc.G) { return; }
  if (pc.stop_on_eos != 0u && stopped[0] != 0u) { return; }
  generated_tokens[index] = next_token[0];
  generated_length[0] = index + 1u;
  if (pc.stop_on_eos != 0u && done[0] != 0u) { stopped[0] = 1u; }
}
""".lstrip(),
)

TOKEN_STORE_EOS_F32 = ShaderVariant(
    name="token_store_eos_f32",
    family=TOKEN_STORE_F32.family,
    contract=ShaderContract(
        class_name="TokenStoreEosF32Program",
        shader_name="token_store_eos_f32",
        fields=TOKEN_STORE_F32.contract.fields,
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("G", PushConstantType.UINT32, 0, "G"),
                PushConstantFieldSpec("stop_on_eos", PushConstantType.UINT32, 4, 1),
            ),
        ),
        dispatch=TOKEN_STORE_F32.contract.dispatch,
    ),
    execution_requirements=TOKEN_STORE_F32.execution_requirements,
    source=TOKEN_STORE_F32.source,
)
