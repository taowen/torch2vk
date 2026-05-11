"""Hand-written fused RoPE shader for optimized OmniVoice."""

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


OMNIVOICE_ROTARY_FUSED_F32 = ShaderVariant(
    name="omnivoice_rotary_fused_f32",
    family="optimized_omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceRotaryFusedF32Program",
        shader_name="omnivoice_rotary_fused_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("B", "N", "S", "D")),
            ),
            TensorFieldSpec(
                name="cos",
                io_kind=IOKind.INPUT,
                role="cos",
                contract=TensorContract(dtype="float32", shape=("B", 1, "S", "D")),
            ),
            TensorFieldSpec(
                name="sin",
                io_kind=IOKind.INPUT,
                role="sin",
                contract=TensorContract(dtype="float32", shape=("B", 1, "S", "D")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=("B", "N", "S", "D")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                PushConstantFieldSpec("N", PushConstantType.UINT32, 4, "N"),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 8, "S"),
                PushConstantFieldSpec("D", PushConstantType.UINT32, 12, "D"),
            ),
        ),
        dispatch=(ceil_div(mul(mul(mul("B", "N"), "S"), "D"), 256), 1, 1),
    ),
    source="""\
#version 450

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly CosBuffer { float cos_values[]; };
layout(set = 0, binding = 2) buffer restrict readonly SinBuffer { float sin_values[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint B; uint N; uint S; uint D; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.N * pc.S * pc.D;
    if (idx >= total) {
        return;
    }

    const uint d = idx % pc.D;
    const uint seq_index = idx / pc.D;
    const uint s = seq_index % pc.S;
    const uint head_index = seq_index / pc.S;
    const uint b = head_index / pc.N;
    const uint half_d = pc.D >> 1u;
    const uint rotated_d = (d < half_d) ? (d + half_d) : (d - half_d);
    const float rotated = (d < half_d) ? -x[idx + half_d] : x[idx - half_d];
    const uint rope_index = (b * pc.S + s) * pc.D + d;

    output_values[idx] = fma(x[idx], cos_values[rope_index], rotated * sin_values[rope_index]);
}
""",
)
