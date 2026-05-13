"""Fused RMSNorm + weight multiply for prefill float32 activations."""

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
    mul,
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
    SubgroupRequirements,
)


RMS_NORM_MUL_F32 = ShaderVariant(
    name="rms_norm_mul_f32",
    family="quantized_qwen3",
    contract=ShaderContract(
        class_name="QuantizedQwen3RmsNormMulF32Program",
        shader_name="rms_norm_mul_f32",
        fields=(
            TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("B", "T", "H"))),
            TensorFieldSpec("weight", IOKind.INPUT, "weight", TensorContract(dtype="float32", shape=("H",))),
            TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=("B", "T", "H"))),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("ROWS", PushConstantType.UINT32, 0, mul("B", "T")),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
            ),
        ),
        dispatch=(mul("B", "T"), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
    ),
    source="""\
#version 450

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };

layout(push_constant) uniform PushConstants { uint ROWS; uint H; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared float row_sum[4];

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint lane = gl_SubgroupInvocationID;
    const uint subgroup = gl_SubgroupID;
    const uint tid = gl_LocalInvocationID.x;
    const uint base = row * pc.H;

    float partial = 0.0;
    for (uint h = tid; h < pc.H; h += 256u) {
        const float v = x[base + h];
        partial = fma(v, v, partial);
    }

    const float subgroup_sum = subgroupAdd(partial);
    if (lane == 0u) {
        row_sum[subgroup] = subgroup_sum;
    }
    barrier();

    if (subgroup == 0u) {
        const float v = lane < 4u ? row_sum[lane] : 0.0;
        const float total = subgroupAdd(v);
        if (lane == 0u) {
            row_sum[0] = total;
        }
    }
    barrier();

    const float inv_rms = inversesqrt(row_sum[0] / float(pc.H) + 0.000001);
    for (uint h = tid; h < pc.H; h += 256u) {
        output_values[base + h] = x[base + h] * inv_rms * weight[h];
    }
}
""",
)
