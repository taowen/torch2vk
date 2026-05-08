"""Qwen3-ASR SwiGLU activation shader: silu(gate) * up."""

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


QWEN3_ASR_TEXT_SWIGLU_F32 = ShaderVariant(
    name="qwen3_asr_text_swiglu_f32",
    family="qwen3_asr.text",
    contract=ShaderContract(
        class_name="Qwen3AsrTextSwigluF32Program",
        shader_name="qwen3_asr_text_swiglu_f32",
        fields=(
            TensorFieldSpec(
                name="gate",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=(1, "T", "I")),
            ),
            TensorFieldSpec(
                name="up",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=(1, "T", "I")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=(1, "T", "I")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, mul("T", "I")),),
        ),
        dispatch=(ceil_div(mul("T", "I"), 256), 1, 1),
    ),
    source="""
#version 450

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly GateBuffer {
    float gate_values[];
};

layout(set = 0, binding = 1) buffer restrict readonly UpBuffer {
    float up_values[];
};

layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer {
    float output_values[];
};

layout(push_constant) uniform PushConstants {
    uint N;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index < pc.N) {
        const float g = gate_values[index];
        const float silu_g = g / (1.0 + exp(-g));
        output_values[index] = silu_g * up_values[index];
    }
}
""".lstrip(),
)
