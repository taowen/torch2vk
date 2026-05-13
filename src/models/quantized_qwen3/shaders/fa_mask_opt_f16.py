"""llama.cpp-style flash-attention mask block classifier."""

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


FA_MASK_OPT_F16 = ShaderVariant(
    name="fa_mask_opt_f16",
    family="quantized_qwen3",
    contract=ShaderContract(
        class_name="QuantizedQwen3FaMaskOptF16Program",
        shader_name="fa_mask_opt_f16",
        fields=(
            TensorFieldSpec(
                "mask",
                IOKind.INPUT,
                "mask",
                TensorContract(dtype="float16", shape=("T", "S")),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="uint32", shape=("W", "R")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=36,
            fields=(
                PushConstantFieldSpec("S", PushConstantType.UINT32, 0, "S"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
                PushConstantFieldSpec("B", PushConstantType.UINT32, 8, 1),
                PushConstantFieldSpec("mask_row_stride", PushConstantType.UINT32, 12, "S"),
                PushConstantFieldSpec("mask_batch_stride", PushConstantType.UINT32, 16, mul("S", "T")),
                PushConstantFieldSpec("mask_outer_stride", PushConstantType.UINT32, 20, mul("S", "T")),
                PushConstantFieldSpec("output_row_stride", PushConstantType.UINT32, 24, "W"),
                PushConstantFieldSpec("output_batch_stride", PushConstantType.UINT32, 28, mul("W", "R")),
                PushConstantFieldSpec("output_outer_stride", PushConstantType.UINT32, 32, mul("W", "R")),
            ),
        ),
        dispatch=("W", "R", 1),
    ),
    specialization_constants=(
        (0, 128),
        (1, 2),
        (2, 16),
        (3, 64),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        require_storage_buffer_16bit_access=True,
    ),
    source="""\
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : enable
layout(std430) buffer;

layout(constant_id = 0) const uint BLOCK_SIZE = 128;
layout(constant_id = 1) const uint NUM_SUBGROUPS = 2;
layout(constant_id = 2) const uint Br = 16;
layout(constant_id = 3) const uint Bc = 64;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer restrict readonly MaskBuffer { float16_t mask_values[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { uint output_values[]; };

layout(push_constant) uniform PushConstants {
    uint S;
    uint T;
    uint B;
    uint mask_row_stride;
    uint mask_batch_stride;
    uint mask_outer_stride;
    uint output_row_stride;
    uint output_batch_stride;
    uint output_outer_stride;
} pc;

const float FLT_MAX_OVER_2 = 1.7014117e38;
const uint MASK_OPT_ALL_NEG_INF = 1;
const uint MASK_OPT_ALL_ZERO = 2;

shared float min_shared[NUM_SUBGROUPS];
shared float max_shared[NUM_SUBGROUPS];

void main() {
    const uint tid = gl_LocalInvocationIndex;
    const uint word_index = gl_WorkGroupID.x;
    const uint row_block = gl_WorkGroupID.y;

    uint result = 0;
    for (uint block_x = 0; block_x < 16; ++block_x) {
        float min_value = FLT_MAX_OVER_2;
        float max_value = -FLT_MAX_OVER_2;

        for (uint idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            const uint col = (idx % Bc) + (word_index * 16 + block_x) * Bc;
            const uint row = (idx / Bc) + row_block * Br;
            if (col < pc.S && row < pc.T) {
                const float value = float(mask_values[row * pc.mask_row_stride + col]);
                min_value = min(min_value, value);
                max_value = max(max_value, value);
            }
        }

        min_value = subgroupMin(min_value);
        max_value = subgroupMax(max_value);
        if (gl_SubgroupInvocationID == 0) {
            min_shared[gl_SubgroupID] = min_value;
            max_shared[gl_SubgroupID] = max_value;
        }
        barrier();

        if (tid == 0) {
            for (uint subgroup = 0; subgroup < NUM_SUBGROUPS; ++subgroup) {
                min_value = min(min_value, min_shared[subgroup]);
                max_value = max(max_value, max_shared[subgroup]);
            }
            if (max_value <= -FLT_MAX_OVER_2) {
                result |= MASK_OPT_ALL_NEG_INF << (2 * block_x);
            }
            if (min_value == 0.0 && max_value == 0.0) {
                result |= MASK_OPT_ALL_ZERO << (2 * block_x);
            }
        }
        barrier();
    }

    if (tid == 0) {
        output_values[word_index + row_block * pc.output_row_stride] = result;
    }
}
""",
)
