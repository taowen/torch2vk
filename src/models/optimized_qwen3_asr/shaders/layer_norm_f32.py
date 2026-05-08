"""Qwen3-ASR row-wise LayerNorm shader with BF16 affine parameters."""

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
)


QWEN3_ASR_LAYER_NORM_F32 = ShaderVariant(
    name="qwen3_asr_layer_norm_f32",
    family="qwen3_asr.audio_tower",
    contract=ShaderContract(
        class_name="Qwen3AsrLayerNormF32Program",
        shader_name="qwen3_asr_layer_norm_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("M", "H")),
            ),
            TensorFieldSpec(
                name="weight",
                io_kind=IOKind.INPUT,
                role="weight",
                contract=TensorContract(dtype="bfloat16", shape=("H",)),
            ),
            TensorFieldSpec(
                name="bias",
                io_kind=IOKind.INPUT,
                role="bias",
                contract=TensorContract(dtype="bfloat16", shape=("H",)),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=("M", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("M", PushConstantType.UINT32, 0, "M"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
                PushConstantFieldSpec("eps", PushConstantType.FLOAT32, 8, 1e-5),
            ),
        ),
        dispatch=("M", 1, 1),
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
    uint M;
    uint H;
    float eps;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared float partial_sum[256];
shared float partial_sumsq[256];

float bf16_to_f32(uint16_t value) {
    return uintBitsToFloat(uint(value) << 16);
}

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    if (row >= pc.M) {
        return;
    }

    float sum = 0.0;
    float sumsq = 0.0;
    for (uint h = tid; h < pc.H; h += 256u) {
        const float v = x[row * pc.H + h];
        sum += v;
        sumsq += v * v;
    }
    partial_sum[tid] = sum;
    partial_sumsq[tid] = sumsq;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        barrier();
    }

    const float mean = partial_sum[0] / float(pc.H);
    const float variance = max(partial_sumsq[0] / float(pc.H) - mean * mean, 0.0);
    const float inv_std = inversesqrt(variance + pc.eps);
    for (uint h = tid; h < pc.H; h += 256u) {
        const float normalized = (x[row * pc.H + h] - mean) * inv_std;
        output_values[row * pc.H + h] = normalized * bf16_to_f32(weight[h]) + bf16_to_f32(bias[h]);
    }
}
""".lstrip(),
)
