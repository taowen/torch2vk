"""Qwen3-ASR compact padded CNN tokens into the encoder sequence."""

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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements


QWEN3_ASR_COMPACT_AFTER_CNN_F32 = ShaderVariant(
    name="qwen3_asr_compact_after_cnn_f32",
    family="qwen3_asr.audio_tower",
    contract=ShaderContract(
        class_name="Qwen3AsrCompactAfterCnnF32Program",
        shader_name="qwen3_asr_compact_after_cnn_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("C", "T", "H")),
            ),
            TensorFieldSpec(
                name="feature_lens",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="int64", shape=(1,)),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=("M", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=20,
            fields=(
                PushConstantFieldSpec("C", PushConstantType.UINT32, 0, "C"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 8, "H"),
                PushConstantFieldSpec("M", PushConstantType.UINT32, 12, "M"),
                PushConstantFieldSpec("N", PushConstantType.UINT32, 16, mul("M", "H")),
            ),
        ),
        dispatch=(ceil_div(mul("M", "H"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer {
    float x[];
};

layout(set = 0, binding = 1) buffer restrict readonly FeatureLensBuffer {
    int64_t feature_lens[];
};

layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer {
    float output_values[];
};

layout(push_constant) uniform PushConstants {
    uint C;
    uint T;
    uint H;
    uint M;
    uint N;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uint ceil_div_u(uint x, uint y) {
    return (x + y - 1u) / y;
}

uint aftercnn_len(uint length) {
    return ceil_div_u(ceil_div_u(ceil_div_u(length, 2u), 2u), 2u);
}

uint chunk_len(uint feature_len, uint chunk) {
    const uint start = chunk * 100u;
    if (start >= feature_len) {
        return 0u;
    }
    return min(100u, feature_len - start);
}

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= pc.N) {
        return;
    }

    const uint h = index % pc.H;
    const uint row = index / pc.H;
    const uint feature_len = uint(feature_lens[0]);
    uint base = 0u;
    for (uint chunk = 0u; chunk < pc.C; ++chunk) {
        const uint valid_steps = aftercnn_len(chunk_len(feature_len, chunk));
        if (row < base + valid_steps) {
            const uint t = row - base;
            output_values[index] = x[(chunk * pc.T + t) * pc.H + h];
            return;
        }
        base += valid_steps;
    }
    output_values[index] = 0.0;
}
""".lstrip(),
)
