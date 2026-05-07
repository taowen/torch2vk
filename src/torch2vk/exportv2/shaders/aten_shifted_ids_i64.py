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

ATEN_SHIFTED_IDS_I64 = ShaderVariant(
    name="aten_shifted_ids_i64",
    family="aten",
    contract=ShaderContract(
        class_name="AtenShiftedIdsI64Program",
        shader_name="aten_shifted_ids_i64",
        fields=(
            TensorFieldSpec(
                "input_ids",
                IOKind.INPUT,
                "input_ids",
                TensorContract(dtype="int64", shape=("B", "C", "T")),
            ),
            TensorFieldSpec(
                "audio_mask",
                IOKind.INPUT,
                "audio_mask",
                TensorContract(dtype="bool", shape=("B", "T")),
            ),
            TensorFieldSpec(
                "codebook_layer_offsets",
                IOKind.INPUT,
                "codebook_layer_offsets",
                TensorContract(dtype="int64", shape=(1, "C", 1)),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="int64", shape=("B", "C", "T")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("C", PushConstantType.UINT32, 0, "C"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
            ),
        ),
        dispatch=(ceil_div("T", 256), "C", "B"),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly InputIdsBuffer { int64_t input_ids[]; };
layout(set = 0, binding = 1) buffer restrict readonly AudioMaskBuffer { bool audio_mask[]; };
layout(set = 0, binding = 2) buffer restrict readonly OffsetsBuffer { int64_t offsets[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { int64_t output_values[]; };
layout(push_constant) uniform PushConstants { uint C; uint T; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
  const uint token = gl_GlobalInvocationID.x;
  const uint codebook = gl_GlobalInvocationID.y;
  const uint batch = gl_GlobalInvocationID.z;
  if (token >= pc.T) { return; }
  const uint idx = (batch * pc.C + codebook) * pc.T + token;
  const int64_t mask = audio_mask[batch * pc.T + token] ? int64_t(1) : int64_t(0);
  output_values[idx] = input_ids[idx] * mask + offsets[codebook];
}
""".lstrip(),
)
