"""Qwen3-ASR single-audio feature pad/copy shader."""

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


QWEN3_ASR_PAD_FEATURE_F32 = ShaderVariant(
    name="qwen3_asr_pad_feature_f32",
    family="qwen3_asr.audio_tower",
    contract=ShaderContract(
        class_name="Qwen3AsrPadFeatureF32Program",
        shader_name="qwen3_asr_pad_feature_f32",
        fields=(
            TensorFieldSpec(
                name="input_features",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("H", "W")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=(1, 1, "H", "W")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, mul("H", "W")),),
        ),
        dispatch=(ceil_div(mul("H", "W"), 256), 1, 1),
    ),
    source="""
#version 450

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly InputBuffer {
    float input_features[];
};

layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer {
    float output_values[];
};

layout(push_constant) uniform PushConstants {
    uint N;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index < pc.N) {
        output_values[index] = input_features[index];
    }
}
""".lstrip(),
)
