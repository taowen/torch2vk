"""Hand-written optimized decode RMSNorm shader."""

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
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
    SubgroupRequirements,
)


DECODE_RMS_NORM_F32 = ShaderVariant(
    name="decode_rms_norm_f32",
    family="optimized_qwen3_asr",
    contract=ShaderContract(
        class_name="OptimizedDecodeRmsNormProgram",
        shader_name="decode_rms_norm_f32",
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
            size=12,
            fields=(
                PushConstantFieldSpec("ROWS", PushConstantType.UINT32, 0, 1),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, 1024),
                PushConstantFieldSpec("EPS", PushConstantType.FLOAT32, 8, 1e-6),
            ),
        ),
        dispatch=(1, 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        require_storage_buffer_16bit_access=True,
    ),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };

layout(push_constant) uniform PushConstants { uint ROWS; uint H; float EPS; } pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint lane = gl_SubgroupInvocationID;
    if (row >= pc.ROWS) { return; }

    const uint row_offset = row * pc.H;
    float sum_sq = 0.0;
    for (uint h = lane; h < pc.H; h += 64u) {
        const float value = float(x[row_offset + h]);
        sum_sq = fma(value, value, sum_sq);
    }
    sum_sq = subgroupAdd(sum_sq);
    const float scale = inversesqrt(sum_sq / float(pc.H) + pc.EPS);

    for (uint h = lane; h < pc.H; h += 64u) {
        output_values[row_offset + h] = float16_t(float(x[row_offset + h]) * scale * weight[h]);
    }
}
""",
)
