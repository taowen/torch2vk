"""Generated OmniVoice aten shifted ids i64 shader scaffold."""

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


OMNIVOICE_ATEN_SHIFTED_IDS_I64 = ShaderVariant(
    name="omnivoice_aten_shifted_ids_i64",
    family="omnivoice.text",
    contract=ShaderContract(
        class_name="OmniVoiceAtenShiftedIdsI64Program",
        shader_name="omnivoice_aten_shifted_ids_i64",
        fields=(
            TensorFieldSpec(
                name="input_ids",
                io_kind=IOKind.INPUT,
                role="input_ids",
                contract=TensorContract(dtype="int64", shape=("B", "C", "T")),
            ),
            TensorFieldSpec(
                name="audio_mask",
                io_kind=IOKind.INPUT,
                role="audio_mask",
                contract=TensorContract(dtype="uint32", shape=("B", "T")),
            ),
            TensorFieldSpec(
                name="codebook_layer_offsets",
                io_kind=IOKind.INPUT,
                role="codebook_layer_offsets",
                contract=TensorContract(dtype="int64", shape=(1, "C", 1)),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="int64", shape=("B", "C", "T")),
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

layout(set = 0, binding = 0) buffer restrict readonly InputIdsBuffer {
    int64_t input_ids[];
};

layout(set = 0, binding = 1) buffer restrict readonly AudioMaskBuffer {
    uint audio_mask[];
};

layout(set = 0, binding = 2) buffer restrict readonly OffsetsBuffer {
    int64_t offsets[];
};

layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer {
    int64_t output_values[];
};

layout(push_constant) uniform PushConstants {
    uint C;
    uint T;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint token = gl_GlobalInvocationID.x;
    const uint codebook = gl_GlobalInvocationID.y;
    const uint batch = gl_GlobalInvocationID.z;
    if (token >= pc.T) {
        return;
    }
    const uint idx = (batch * pc.C + codebook) * pc.T + token;
    const int64_t mask = audio_mask[batch * pc.T + token] != 0u ? int64_t(1) : int64_t(0);
    output_values[idx] = input_ids[idx] * mask + offsets[codebook];
}
""".lstrip(),
)

