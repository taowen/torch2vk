"""Fused RMSNorm and RoPE with token-major float32 output."""

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


RMS_NORM_ROPE_TOKEN_F32 = ShaderVariant(
    name="rms_norm_rope_token_f32",
    family="quantized_qwen3",
    contract=ShaderContract(
        class_name="QuantizedQwen3RmsNormRopeTokenF32Program",
        shader_name="rms_norm_rope_token_f32",
        fields=(
            TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("B", "T", "G", "D"))),
            TensorFieldSpec("weight", IOKind.INPUT, "weight", TensorContract(dtype="float32", shape=("D",))),
            TensorFieldSpec("cos", IOKind.INPUT, "input", TensorContract(dtype="float16", shape=("B", "T", "D"))),
            TensorFieldSpec("sin", IOKind.INPUT, "input", TensorContract(dtype="float16", shape=("B", "T", "D"))),
            TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=("B", "T", "G", "D"))),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec("ROWS", PushConstantType.UINT32, 0, mul(mul("B", "T"), "G")),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
                PushConstantFieldSpec("G", PushConstantType.UINT32, 8, "G"),
                PushConstantFieldSpec("D", PushConstantType.UINT32, 12, "D"),
            ),
        ),
        dispatch=(mul(mul("B", "T"), "G"), 1, 1),
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
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 2) buffer restrict readonly CosBuffer { float16_t cos_values[]; };
layout(set = 0, binding = 3) buffer restrict readonly SinBuffer { float16_t sin_values[]; };
layout(set = 0, binding = 4) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint ROWS; uint T; uint G; uint D; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float row_sum[4];
void main() {
    const uint row = gl_WorkGroupID.x;
    const uint lane = gl_SubgroupInvocationID;
    const uint subgroup = gl_SubgroupID;
    const uint tid = gl_LocalInvocationID.x;
    const uint d = tid;
    const uint g = row % pc.G;
    const uint t = (row / pc.G) % pc.T;
    const uint b = row / (pc.G * pc.T);
    const uint x_base = row * pc.D;
    float partial = 0.0;
    for (uint h = tid; h < pc.D; h += 256u) {
        const float v = x[x_base + h];
        partial = fma(v, v, partial);
    }
    const float subgroup_sum = subgroupAdd(partial);
    if (lane == 0u) { row_sum[subgroup] = subgroup_sum; }
    barrier();
    if (subgroup == 0u) {
        const float v = lane < 4u ? row_sum[lane] : 0.0;
        const float total = subgroupAdd(v);
        if (lane == 0u) { row_sum[0] = total; }
    }
    barrier();
    if (d >= pc.D) { return; }
    const uint half_d = pc.D >> 1u;
    const uint rotated_d = d < half_d ? d + half_d : d - half_d;
    const float sign_value = d < half_d ? -1.0 : 1.0;
    const float inv_rms = inversesqrt(row_sum[0] / float(pc.D) + 0.000001);
    const float norm_value = x[x_base + d] * inv_rms * weight[d];
    const float norm_rotated = sign_value * x[x_base + rotated_d] * inv_rms * weight[rotated_d];
    const uint rope_idx = (b * pc.T + t) * pc.D + d;
    output_values[((b * pc.T + t) * pc.G + g) * pc.D + d] =
        norm_value * float(cos_values[rope_idx]) + norm_rotated * float(sin_values[rope_idx]);
}
""",
)
