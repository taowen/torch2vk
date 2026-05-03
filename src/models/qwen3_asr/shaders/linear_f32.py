"""Qwen3-ASR BF16-weight linear shaders with workgroup K reduction."""

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


_LINEAR_SOURCE = """
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

{gelu_define}

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer {{
    float x[];
}};

layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer {{
    uint16_t weight[];
}};

layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer {{
    uint16_t bias[];
}};

layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer {{
    float output_values[];
}};

layout(push_constant) uniform PushConstants {{
    uint M;
    uint K;
    uint N;
}} pc;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

const uint TILE_M = 16u;
const uint TILE_N = 16u;
const uint TILE_K = 32u;

shared float tile_x[16 * 32];
shared float tile_w[32 * 16];

float bf16_to_f32(uint16_t value) {{
    return uintBitsToFloat(uint(value) << 16);
}}

float gelu_erf(float value) {{
    const float p_erf = 0.3275911;
    const float a1_erf = 0.254829592;
    const float a2_erf = -0.284496736;
    const float a3_erf = 1.421413741;
    const float a4_erf = -1.453152027;
    const float a5_erf = 1.061405429;
    const float sqrt_2_inv = 0.7071067811865475;

    const float scaled = value * sqrt_2_inv;
    const float sign_x = sign(scaled);
    const float abs_x = abs(scaled);
    const float t = 1.0 / (1.0 + p_erf * abs_x);
    const float y = 1.0 - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-abs_x * abs_x);
    return 0.5 * value * (1.0 + sign_x * y);
}}

void main() {{
    const uint local_col = gl_LocalInvocationID.x;
    const uint local_row = gl_LocalInvocationID.y;
    const uint lane = local_row * TILE_N + local_col;
    const uint row = gl_WorkGroupID.x * TILE_M + local_row;
    const uint col = gl_WorkGroupID.y * TILE_N + local_col;

    float acc = 0.0;
    for (uint k0 = 0u; k0 < pc.K; k0 += TILE_K) {{
        for (uint i = lane; i < TILE_M * TILE_K; i += TILE_M * TILE_N) {{
            const uint tile_row = i / TILE_K;
            const uint tile_k = i - tile_row * TILE_K;
            const uint global_row = gl_WorkGroupID.x * TILE_M + tile_row;
            const uint global_k = k0 + tile_k;
            tile_x[i] = global_row < pc.M && global_k < pc.K ? x[global_row * pc.K + global_k] : 0.0;
        }}
        for (uint i = lane; i < TILE_K * TILE_N; i += TILE_M * TILE_N) {{
            const uint tile_k = i / TILE_N;
            const uint tile_col = i - tile_k * TILE_N;
            const uint global_k = k0 + tile_k;
            const uint global_col = gl_WorkGroupID.y * TILE_N + tile_col;
            tile_w[i] = global_col < pc.N && global_k < pc.K ? bf16_to_f32(weight[global_col * pc.K + global_k]) : 0.0;
        }}
        barrier();

        [[unroll]] for (uint k = 0u; k < TILE_K; ++k) {{
            acc += tile_x[local_row * TILE_K + k] * tile_w[k * TILE_N + local_col];
        }}
        barrier();
    }}

    if (row < pc.M && col < pc.N) {{
        float value = acc + bf16_to_f32(bias[col]);
#if APPLY_GELU
        value = gelu_erf(value);
#endif
        output_values[row * pc.N + col] = value;
    }}
}}
""".lstrip()


def _linear_variant(*, name: str, class_name: str, apply_gelu: bool) -> ShaderVariant:
    return ShaderVariant(
        name=name,
        family="qwen3_asr.audio_tower",
        contract=ShaderContract(
            class_name=class_name,
            shader_name=name,
            fields=(
                TensorFieldSpec(
                    name="x",
                    io_kind=IOKind.INPUT,
                    role="input",
                    contract=TensorContract(dtype="float32", shape=("M", "K")),
                ),
                TensorFieldSpec(
                    name="weight",
                    io_kind=IOKind.INPUT,
                    role="weight",
                    contract=TensorContract(dtype="bfloat16", shape=("N", "K")),
                ),
                TensorFieldSpec(
                    name="bias",
                    io_kind=IOKind.INPUT,
                    role="bias",
                    contract=TensorContract(dtype="bfloat16", shape=("N",)),
                ),
                TensorFieldSpec(
                    name="output",
                    io_kind=IOKind.OUTPUT,
                    role="output",
                    contract=TensorContract(dtype="float32", shape=("M", "N")),
                ),
            ),
            push_constants=PushConstantSpec(
                size=12,
                fields=(
                    PushConstantFieldSpec("M", PushConstantType.UINT32, 0, "M"),
                    PushConstantFieldSpec("K", PushConstantType.UINT32, 4, "K"),
                    PushConstantFieldSpec("N", PushConstantType.UINT32, 8, "N"),
                ),
            ),
            dispatch=(ceil_div("M", 16), ceil_div("N", 16), 1),
        ),
        execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
        source=_LINEAR_SOURCE.format(gelu_define="#define APPLY_GELU 1" if apply_gelu else "#define APPLY_GELU 0"),
    )


QWEN3_ASR_LINEAR_F32 = _linear_variant(
    name="qwen3_asr_linear_f32",
    class_name="Qwen3AsrLinearF32Program",
    apply_gelu=False,
)

QWEN3_ASR_LINEAR_GELU_F32 = _linear_variant(
    name="qwen3_asr_linear_gelu_f32",
    class_name="Qwen3AsrLinearGeluF32Program",
    apply_gelu=True,
)
