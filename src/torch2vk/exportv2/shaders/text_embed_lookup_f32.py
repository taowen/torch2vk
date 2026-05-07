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
    ceil_div,
)

TEXT_EMBED_LOOKUP_F32 = ShaderVariant(
    name="text_embed_lookup_f32",
    family="text",
    contract=ShaderContract(
        class_name="TextEmbedLookupF32Program",
        shader_name="text_embed_lookup_f32",
        fields=(
            TensorFieldSpec(
                "input_ids", IOKind.INPUT, "input", TensorContract(dtype="int64", shape=(1, 1))
            ),
            TensorFieldSpec(
                "embed_tokens_weight",
                IOKind.INPUT,
                "weight",
                TensorContract(dtype="bfloat16", shape=("V", "H")),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float32", shape=(1, 1, "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("H", PushConstantType.UINT32, 0, "H"),
                PushConstantFieldSpec("V", PushConstantType.UINT32, 4, "V"),
            ),
        ),
        dispatch=(ceil_div("H", 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly InputIdsBuffer { int input_ids[]; };
layout(set = 0, binding = 1) buffer restrict readonly EmbedBuffer { uint16_t embed_tokens[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint H; uint V; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
float bf16_to_f32(uint16_t value) { return uintBitsToFloat(uint(value) << 16); }
void main() {
  const uint h = gl_GlobalInvocationID.x;
  if (h >= pc.H) { return; }
  const uint token_id = uint(input_ids[0]);
  output_values[h] = bf16_to_f32(embed_tokens[token_id * pc.H + h]);
}
""".lstrip(),
)
