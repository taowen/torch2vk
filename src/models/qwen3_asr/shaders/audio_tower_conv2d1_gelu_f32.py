"""Qwen3-ASR first audio tower shader: conv2d1 + bias + GELU."""

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
    ShaderExecutionRequirements,
)


QWEN3_ASR_AUDIO_TOWER_CONV2D1_GELU_F32 = ShaderVariant(
    name="qwen3_asr_audio_tower_conv2d1_gelu_f32",
    family="qwen3_asr.audio_tower",
    contract=ShaderContract(
        class_name="Qwen3AsrAudioTowerConv2d1GeluF32Program",
        shader_name="qwen3_asr_audio_tower_conv2d1_gelu_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("N", "IC", "IH", "IW")),
            ),
            TensorFieldSpec(
                name="weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(dtype="bfloat16", shape=("OC", "IC", "KH", "KW")),
            ),
            TensorFieldSpec(
                name="bias",
                io_kind=IOKind.INPUT,
                role="bias",
                contract=TensorContract(dtype="bfloat16", shape=("OC",)),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=("N", "OC", "OH", "OW")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=36,
            fields=(
                PushConstantFieldSpec("N", PushConstantType.UINT32, 0, "N"),
                PushConstantFieldSpec("IC", PushConstantType.UINT32, 4, "IC"),
                PushConstantFieldSpec("IH", PushConstantType.UINT32, 8, "IH"),
                PushConstantFieldSpec("IW", PushConstantType.UINT32, 12, "IW"),
                PushConstantFieldSpec("OC", PushConstantType.UINT32, 16, "OC"),
                PushConstantFieldSpec("OH", PushConstantType.UINT32, 20, "OH"),
                PushConstantFieldSpec("OW", PushConstantType.UINT32, 24, "OW"),
                PushConstantFieldSpec("KH", PushConstantType.UINT32, 28, "KH"),
                PushConstantFieldSpec("KW", PushConstantType.UINT32, 32, "KW"),
            ),
        ),
        dispatch=(ceil_div(mul(mul(mul("N", "OC"), "OH"), "OW"), 256), 1, 1),
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

layout(set = 0, binding = 2) buffer restrict readonly BiasBuffer {
    uint16_t bias[];
};

layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer {
    float output_values[];
};

layout(push_constant) uniform PushConstants {
    uint N;
    uint IC;
    uint IH;
    uint IW;
    uint OC;
    uint OH;
    uint OW;
    uint KH;
    uint KW;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

float bf16_to_f32(uint16_t value) {
    return uintBitsToFloat(uint(value) << 16);
}

float gelu_erf(float value) {
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
}

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint total = pc.N * pc.OC * pc.OH * pc.OW;
    if (index >= total) {
        return;
    }

    const uint ow = index % pc.OW;
    const uint oh = (index / pc.OW) % pc.OH;
    const uint oc = (index / (pc.OW * pc.OH)) % pc.OC;
    const uint n = index / (pc.OW * pc.OH * pc.OC);

    float acc = bf16_to_f32(bias[oc]);
    for (uint ic = 0; ic < pc.IC; ++ic) {
        for (uint kh = 0; kh < pc.KH; ++kh) {
            const int ih = int(oh * 2u + kh) - 1;
            if (ih < 0 || ih >= int(pc.IH)) {
                continue;
            }
            for (uint kw = 0; kw < pc.KW; ++kw) {
                const int iw = int(ow * 2u + kw) - 1;
                if (iw < 0 || iw >= int(pc.IW)) {
                    continue;
                }
                const uint x_index = ((n * pc.IC + ic) * pc.IH + uint(ih)) * pc.IW + uint(iw);
                const uint w_index = ((oc * pc.IC + ic) * pc.KH + kh) * pc.KW + kw;
                acc += x[x_index] * bf16_to_f32(weight[w_index]);
            }
        }
    }
    output_values[index] = gelu_erf(acc);
}
""".lstrip(),
)
