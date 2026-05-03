"""Qwen3-ASR audio tower shader: conv_out linear + sinusoidal position add."""

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
    mul,
)


QWEN3_ASR_CONV_OUT_ADD_POSITION_F32 = ShaderVariant(
    name="qwen3_asr_audio_tower_conv_out_add_position_f32",
    family="qwen3_asr.audio_tower",
    contract=ShaderContract(
        class_name="Qwen3AsrAudioTowerConvOutAddPositionF32Program",
        shader_name="qwen3_asr_audio_tower_conv_out_add_position_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("N", "C", "F", "T")),
            ),
            TensorFieldSpec(
                name="weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(dtype="bfloat16", shape=("H", mul("C", "F"))),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=(mul("N", "T"), "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=24,
            fields=(
                PushConstantFieldSpec("N", PushConstantType.UINT32, 0, "N"),
                PushConstantFieldSpec("C", PushConstantType.UINT32, 4, "C"),
                PushConstantFieldSpec("F", PushConstantType.UINT32, 8, "F"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 12, "T"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 16, "H"),
                PushConstantFieldSpec("K", PushConstantType.UINT32, 20, mul("C", "F")),
            ),
        ),
        dispatch=(ceil_div(mul(mul("N", "T"), "H"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer {
    float x[];
};

layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer {
    uint16_t weight[];
};

layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer {
    float output_values[];
};

layout(push_constant) uniform PushConstants {
    uint N;
    uint C;
    uint F;
    uint T;
    uint H;
    uint K;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

float bf16_to_f32(uint16_t value) {
    return uintBitsToFloat(uint(value) << 16);
}

float position_value(uint t, uint h) {
    const uint half_channels = pc.H / 2u;
    const float denom = float(max(1u, half_channels - 1u));
    const float log_increment = log(10000.0) / denom;
    const uint local_h = h < half_channels ? h : h - half_channels;
    const float freq = exp(-log_increment * float(local_h));
    const float scaled_time = float(t) * freq;
    return h < half_channels ? sin(scaled_time) : cos(scaled_time);
}

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint total = pc.N * pc.T * pc.H;
    if (index >= total) {
        return;
    }

    const uint h = index % pc.H;
    const uint t = (index / pc.H) % pc.T;
    const uint n = index / (pc.H * pc.T);

    float acc = 0.0;
    for (uint c = 0; c < pc.C; ++c) {
        for (uint f = 0; f < pc.F; ++f) {
            const uint k = c * pc.F + f;
            const uint x_index = ((n * pc.C + c) * pc.F + f) * pc.T + t;
            const uint w_index = h * pc.K + k;
            acc += x[x_index] * bf16_to_f32(weight[w_index]);
        }
    }

    output_values[index] = acc + position_value(t, h);
}
""".lstrip(),
)
