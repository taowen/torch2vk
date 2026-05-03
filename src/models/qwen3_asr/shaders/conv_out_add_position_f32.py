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
        dispatch=(ceil_div(mul("N", "T"), 16), ceil_div("H", 16), 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""
#version 450

#extension GL_EXT_control_flow_attributes : enable
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

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

const uint TILE_M = 16u;
const uint TILE_N = 16u;
const uint TILE_K = 32u;

shared float tile_x[16 * 32];
shared float tile_w[32 * 16];

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
    const uint local_h = gl_LocalInvocationID.x;
    const uint local_row = gl_LocalInvocationID.y;
    const uint lane = local_row * TILE_N + local_h;
    const uint row = gl_WorkGroupID.x * TILE_M + local_row;
    const uint h = gl_WorkGroupID.y * TILE_N + local_h;
    const uint total_rows = pc.N * pc.T;
    const uint n = row / pc.T;
    const uint t = row - n * pc.T;

    float acc = 0.0;
    for (uint k0 = 0u; k0 < pc.K; k0 += TILE_K) {
        for (uint i = lane; i < TILE_M * TILE_K; i += TILE_M * TILE_N) {
            const uint tile_row = i / TILE_K;
            const uint tile_k = i - tile_row * TILE_K;
            const uint global_row = gl_WorkGroupID.x * TILE_M + tile_row;
            const uint global_k = k0 + tile_k;
            const uint load_n = global_row / pc.T;
            const uint load_t = global_row - load_n * pc.T;
            const uint c = global_k / pc.F;
            const uint f = global_k - c * pc.F;
            const uint x_index = ((load_n * pc.C + c) * pc.F + f) * pc.T + load_t;
            tile_x[i] = global_row < total_rows && global_k < pc.K ? x[x_index] : 0.0;
        }
        for (uint i = lane; i < TILE_K * TILE_N; i += TILE_M * TILE_N) {
            const uint tile_k = i / TILE_N;
            const uint tile_h = i - tile_k * TILE_N;
            const uint global_k = k0 + tile_k;
            const uint global_h = gl_WorkGroupID.y * TILE_N + tile_h;
            tile_w[i] = global_h < pc.H && global_k < pc.K ? bf16_to_f32(weight[global_h * pc.K + global_k]) : 0.0;
        }
        barrier();

        [[unroll]] for (uint k = 0u; k < TILE_K; ++k) {
            acc += tile_x[local_row * TILE_K + k] * tile_w[k * TILE_N + local_h];
        }
        barrier();
    }

    if (row < total_rows && h < pc.H) {
        output_values[row * pc.H + h] = acc + position_value(t, h);
    }
}
""".lstrip(),
)
