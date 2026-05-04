"""Qwen3-ASR BF16-weight linear shader without bias for text model."""

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


QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32 = ShaderVariant(
    name="qwen3_asr_text_linear_nobias_f32",
    family="qwen3_asr.text",
    contract=ShaderContract(
        class_name="Qwen3AsrTextLinearNobiasF32Program",
        shader_name="qwen3_asr_text_linear_nobias_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=(1, "T", "K")),
            ),
            TensorFieldSpec(
                name="weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(dtype="bfloat16", shape=("N", "K")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=(1, "T", "N")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("M", PushConstantType.UINT32, 0, "T"),
                PushConstantFieldSpec("K", PushConstantType.UINT32, 4, "K"),
                PushConstantFieldSpec("N", PushConstantType.UINT32, 8, "N"),
            ),
        ),
        dispatch=(ceil_div("T", 16), ceil_div("N", 16), 1),
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
    uint M;
    uint K;
    uint N;
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

void main() {
    const uint local_col = gl_LocalInvocationID.x;
    const uint local_row = gl_LocalInvocationID.y;
    const uint lane = local_row * TILE_N + local_col;
    const uint row = gl_WorkGroupID.x * TILE_M + local_row;
    const uint col = gl_WorkGroupID.y * TILE_N + local_col;

    float acc = 0.0;
    for (uint k0 = 0u; k0 < pc.K; k0 += TILE_K) {
        for (uint i = lane; i < TILE_M * TILE_K; i += TILE_M * TILE_N) {
            const uint tile_row = i / TILE_K;
            const uint tile_k = i - tile_row * TILE_K;
            const uint global_row = gl_WorkGroupID.x * TILE_M + tile_row;
            const uint global_k = k0 + tile_k;
            tile_x[i] = global_row < pc.M && global_k < pc.K ? x[global_row * pc.K + global_k] : 0.0;
        }
        for (uint i = lane; i < TILE_K * TILE_N; i += TILE_M * TILE_N) {
            const uint tile_k = i / TILE_N;
            const uint tile_col = i - tile_k * TILE_N;
            const uint global_k = k0 + tile_k;
            const uint global_col = gl_WorkGroupID.y * TILE_N + tile_col;
            tile_w[i] = global_col < pc.N && global_k < pc.K ? bf16_to_f32(weight[global_col * pc.K + global_k]) : 0.0;
        }
        barrier();

        [[unroll]] for (uint k = 0u; k < TILE_K; ++k) {
            acc += tile_x[local_row * TILE_K + k] * tile_w[k * TILE_N + local_col];
        }
        barrier();
    }

    if (row < pc.M && col < pc.N) {
        output_values[row * pc.N + col] = acc;
    }
}
""".lstrip(),
)
