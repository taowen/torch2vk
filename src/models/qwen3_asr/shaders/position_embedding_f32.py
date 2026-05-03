"""Qwen3-ASR audio sinusoidal position embedding shader."""

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


QWEN3_ASR_POSITION_EMBEDDING_F32 = ShaderVariant(
    name="qwen3_asr_position_embedding_f32",
    family="qwen3_asr.audio_tower",
    contract=ShaderContract(
        class_name="Qwen3AsrPositionEmbeddingF32Program",
        shader_name="qwen3_asr_position_embedding_f32",
        fields=(
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=("T", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("T", PushConstantType.UINT32, 0, "T"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
            ),
        ),
        dispatch=(ceil_div(mul("T", "H"), 256), 1, 1),
    ),
    source="""
#version 450

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer {
    float t_output[];
};

layout(push_constant) uniform PushConstants {
    uint T;
    uint H;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint total = pc.T * pc.H;
    if (index >= total) {
        return;
    }

    const uint half_channels = pc.H / 2u;
    const uint h = index % pc.H;
    const uint s = (index / pc.H) % pc.T;
    const float time = float(s);
    const float denom = float(max(1u, half_channels - 1u));
    const float log_increment = log(10000.0) / denom;

    if (h < half_channels) {
        const float freq = exp(-log_increment * float(h));
        t_output[index] = sin(time * freq);
        return;
    }

    const uint h2 = h - half_channels;
    const float freq = exp(-log_increment * float(h2));
    t_output[index] = cos(time * freq);
}
""".lstrip(),
)
