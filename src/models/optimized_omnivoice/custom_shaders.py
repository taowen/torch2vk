"""Custom optimized OmniVoice shader variants used by export.py."""

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
from torch2vk.vulkan.types import q8_0_halfwords_layout


OMNIVOICE_INPUT_EMBED_Q8_0_F32 = ShaderVariant(
    name="omnivoice_input_embed_q8_0_f32",
    family="optimized_omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceInputEmbedQ8_0Program",
        shader_name="omnivoice_input_embed_q8_0_f32",
        fields=(
            TensorFieldSpec(
                name="text_weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(
                    dtype="uint16",
                    shape=("TV", 544),
                    layout=q8_0_halfwords_layout(logical_k="H"),
                ),
            ),
            TensorFieldSpec(
                name="audio_weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(
                    dtype="uint16",
                    shape=("CV", 544),
                    layout=q8_0_halfwords_layout(logical_k="H"),
                ),
            ),
            TensorFieldSpec(
                name="batch_input_ids",
                io_kind=IOKind.INPUT,
                role="tokens",
                contract=TensorContract(dtype="int64", shape=("B", "C", "S")),
            ),
            TensorFieldSpec(
                name="batch_audio_mask",
                io_kind=IOKind.INPUT,
                role="mask",
                contract=TensorContract(dtype="uint32", shape=("B", "S")),
            ),
            TensorFieldSpec(
                name="hidden_states",
                io_kind=IOKind.OUTPUT,
                role="hidden_states",
                contract=TensorContract(dtype="float32", shape=("B", "S", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=20,
            fields=(
                PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B", dynamic=False),
                PushConstantFieldSpec("C", PushConstantType.UINT32, 4, "C", dynamic=False),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 8, "S", dynamic=False),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 12, "H", dynamic=False),
                PushConstantFieldSpec("V", PushConstantType.UINT32, 16, 1025, dynamic=False),
            ),
        ),
        dispatch=(ceil_div(mul(mul("B", "S"), "H"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        require_shader_int64=True,
        require_storage_buffer_16bit_access=True,
    ),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly TextWeightBuffer {
    uint16_t text_weight[];
};

layout(set = 0, binding = 1) buffer restrict readonly AudioWeightBuffer {
    uint16_t audio_weight[];
};

layout(set = 0, binding = 2) buffer restrict readonly BatchInputIdsBuffer {
    int64_t batch_input_ids[];
};

layout(set = 0, binding = 3) buffer restrict readonly BatchAudioMaskBuffer {
    uint batch_audio_mask[];
};

layout(set = 0, binding = 4) buffer restrict writeonly HiddenStatesBuffer {
    float hidden_states[];
};

layout(push_constant) uniform PushConstants {
    uint B;
    uint C;
    uint S;
    uint H;
    uint V;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

float q8_from_text(uint row, uint h) {
    const uint blocks_per_row = pc.H / 32u;
    const uint block_index = h >> 5u;
    const uint block_half = row * blocks_per_row * 17u + block_index * 17u;
    const float d = unpackHalf2x16(uint(text_weight[block_half])).x;
    const uint local = h & 31u;
    const uint packed = uint(text_weight[block_half + 1u + (local >> 1u)]);
    uint byte_value = ((local & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
    int quant = int(byte_value);
    if (quant >= 128) {
        quant -= 256;
    }
    return d * float(quant);
}

float q8_from_audio(uint row, uint h) {
    const uint blocks_per_row = pc.H / 32u;
    const uint block_index = h >> 5u;
    const uint block_half = row * blocks_per_row * 17u + block_index * 17u;
    const float d = unpackHalf2x16(uint(audio_weight[block_half])).x;
    const uint local = h & 31u;
    const uint packed = uint(audio_weight[block_half + 1u + (local >> 1u)]);
    uint byte_value = ((local & 1u) == 0u) ? (packed & 255u) : (packed >> 8u);
    int quant = int(byte_value);
    if (quant >= 128) {
        quant -= 256;
    }
    return d * float(quant);
}

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.S * pc.H;
    if (idx >= total) {
        return;
    }

    const uint h = idx % pc.H;
    const uint seq_idx = idx / pc.H;
    const uint s = seq_idx % pc.S;
    const uint b = seq_idx / pc.S;

    if (batch_audio_mask[b * pc.S + s] != 0u) {
        float value = 0.0;
        for (uint c = 0u; c < pc.C; ++c) {
            const uint input_offset = (b * pc.C + c) * pc.S + s;
            const uint token = uint(batch_input_ids[input_offset]);
            value += q8_from_audio(c * pc.V + token, h);
        }
        hidden_states[idx] = value;
        return;
    }

    const uint text_offset = (b * pc.C) * pc.S + s;
    const uint text_token = uint(batch_input_ids[text_offset]);
    hidden_states[idx] = q8_from_text(text_token, h);
}
""",
)


RMS_NORM_SOURCE = """\
#version 450

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint rows; uint H; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared float partial[256];

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint lane = gl_LocalInvocationID.x;
    float sum = 0.0;

    for (uint h = lane; h < pc.H; h += 256u) {
        const float value = x[row * pc.H + h];
        sum += value * value;
    }
    partial[lane] = sum;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        barrier();
    }

    const float scale = inversesqrt(partial[0] / float(pc.H) + 0.000001);
    for (uint h = lane; h < pc.H; h += 256u) {
        const uint index = row * pc.H + h;
        output_values[index] = x[index] * scale * weight[h];
    }
}
"""


OMNIVOICE_RMS_NORM_3D_F32 = ShaderVariant(
    name="omnivoice_rms_norm_3d_f32",
    family="optimized_omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceRmsNorm3DF32Program",
        shader_name="omnivoice_rms_norm_3d_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("D0", "D1", "H")),
            ),
            TensorFieldSpec(
                name="weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(dtype="float32", shape=("H",)),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=("D0", "D1", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("rows", PushConstantType.UINT32, 0, mul("D0", "D1")),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
            ),
        ),
        dispatch=(mul("D0", "D1"), 1, 1),
    ),
    source=RMS_NORM_SOURCE,
)


OMNIVOICE_RMS_NORM_4D_F32 = ShaderVariant(
    name="omnivoice_rms_norm_4d_f32",
    family="optimized_omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceRmsNorm4DF32Program",
        shader_name="omnivoice_rms_norm_4d_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("D0", "D1", "D2", "H")),
            ),
            TensorFieldSpec(
                name="weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(dtype="float32", shape=("H",)),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=("D0", "D1", "D2", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("rows", PushConstantType.UINT32, 0, mul(mul("D0", "D1"), "D2")),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
            ),
        ),
        dispatch=(mul(mul("D0", "D1"), "D2"), 1, 1),
    ),
    source=RMS_NORM_SOURCE,
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
    const float rotated = (d < half_d) ? -x[idx + half_d] : x[idx - half_d];
    const uint rope_index = (b * pc.S + s) * pc.D + d;

    output_values[idx] = fma(x[idx], cos_values[rope_index], rotated * sin_values[rope_index]);
}
""",
)


OMNIVOICE_SILU_MUL_F32 = ShaderVariant(
    name="omnivoice_silu_mul_f32",
    family="optimized_omnivoice",
    contract=ShaderContract(
        class_name="OmniVoiceSiluMulF32Program",
        shader_name="omnivoice_silu_mul_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("D0", "D1", "H")),
            ),
            TensorFieldSpec(
                name="y",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("D0", "D1", "H")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=("D0", "D1", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec("N", PushConstantType.UINT32, 0, mul(mul("D0", "D1"), "H")),
            ),
        ),
        dispatch=(ceil_div(mul(mul("D0", "D1"), "H"), 256), 1, 1),
    ),
    source="""\
#version 450

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint N; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) {
        return;
    }
    const float value = x[idx];
    output_values[idx] = (value / (1.0 + exp(-value))) * y[idx];
}
""",
)
