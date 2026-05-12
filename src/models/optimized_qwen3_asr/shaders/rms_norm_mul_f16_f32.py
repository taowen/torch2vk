"""Fused RMSNorm + weight multiply for 3D float16 activations."""

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


RMS_NORM_MUL_F16_F32 = ShaderVariant(
    name="rms_norm_mul_f16_f32",
    family="optimized_qwen3_asr",
    contract=ShaderContract(
        class_name="OptimizedQwen3RmsNormMulF16F32Program",
        shader_name="rms_norm_mul_f16_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float16", shape=("B", "T", "H")),
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
                contract=TensorContract(dtype="float16", shape=("B", "T", "H")),
            ),
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
        require_storage_buffer_16bit_access=True,
    ),
    source="""\
#version 450

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };

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
        const float v = float(x[base + h]);
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
        output_values[base + h] = float16_t(float(x[base + h]) * inv_rms * weight[h]);
    }
}
""",
)
