"""Copy the last sequence position from a prefill hidden-state tensor."""

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
)


SLICE_LAST_TOKEN_F32 = ShaderVariant(
    name="slice_last_token_f32",
    family="quantized_qwen3_asr",
    contract=ShaderContract(
        class_name="SliceLastTokenF32Program",
        shader_name="slice_last_token_f32",
        fields=(
            TensorFieldSpec(
                "x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("B", "S", "H"))
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float32", shape=("B", 1, "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 4, "S"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 8, "H"),
            ),
        ),
        dispatch=(ceil_div("H", 256), "B", 1),
    ),
    source="""\
#version 450

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float out_values[]; };

layout(push_constant) uniform PushConstants { uint B; uint S; uint H; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint h = gl_GlobalInvocationID.x;
    const uint b = gl_GlobalInvocationID.y;
    if (b >= pc.B || h >= pc.H) {
        return;
    }
    out_values[b * pc.H + h] = x[(b * pc.S + (pc.S - 1u)) * pc.H + h];
}
""",
)
