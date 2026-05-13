"""Decode SDPA fused with single-token KV cache write."""

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


SDPA_DECODE_CACHE_WRITE_F16 = ShaderVariant(
    name="sdpa_decode_cache_write_f16",
    family="optimized_qwen3_asr",
    contract=ShaderContract(
        class_name="SdpaDecodeCacheWriteF16Program",
        shader_name="sdpa_decode_cache_write_f16",
        fields=(
            TensorFieldSpec(
                "q",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "NH", "T", "D")),
            ),
            TensorFieldSpec(
                "new_k",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "NK", "T", "D")),
            ),
            TensorFieldSpec(
                "new_v",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "T", "NK", "D")),
            ),
            TensorFieldSpec(
                "k_cache",
                IOKind.INOUT,
                "state",
                TensorContract(dtype="float16", shape=("B", "NK", "S", "D")),
            ),
            TensorFieldSpec(
                "v_cache",
                IOKind.INOUT,
                "state",
                TensorContract(dtype="float16", shape=("B", "NK", "S", "D")),
            ),
            TensorFieldSpec(
                "cache_position",
                IOKind.INPUT,
                "cache_position",
                TensorContract(dtype="int64", shape=("T",)),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float16", shape=("B", "NH", "T", "D")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec("NH", PushConstantType.UINT32, 0, "NH"),
                PushConstantFieldSpec("NK", PushConstantType.UINT32, 4, "NK"),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 8, "S"),
                PushConstantFieldSpec("D", PushConstantType.UINT32, 12, "D"),
            ),
        ),
        dispatch=("NH", 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        require_shader_int64=True,
        require_storage_buffer_16bit_access=True,
    ),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float16_t q[]; };
layout(set = 0, binding = 1) buffer restrict readonly NewKBuffer { float16_t new_k[]; };
layout(set = 0, binding = 2) buffer restrict readonly NewVBuffer { float16_t new_v[]; };
layout(set = 0, binding = 3) buffer restrict KCacheBuffer { float16_t k_cache[]; };
layout(set = 0, binding = 4) buffer restrict VCacheBuffer { float16_t v_cache[]; };
layout(set = 0, binding = 5) buffer restrict readonly CachePositionBuffer { int64_t cache_position[]; };
layout(set = 0, binding = 6) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint NH; uint NK; uint S; uint D; } pc;
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

const float NEG_INF = -3.4028234663852886e38;

shared float subgroup_dot[4];

void main() {
    const uint head = gl_WorkGroupID.x;
    const uint dim = gl_LocalInvocationID.x;
    if (head >= pc.NH) { return; }
    const bool valid_dim = dim < pc.D;

    const uint kv_head = head * pc.NK / pc.NH;
    const uint q_heads_per_kv = pc.NH / pc.NK;
    const bool cache_writer = head == kv_head * q_heads_per_kv;
    const uint current_pos = uint(cache_position[0]);
    const uint cache_head_base = kv_head * pc.S * pc.D;
    const uint new_offset = kv_head * pc.D + dim;
    const float new_k_value = valid_dim ? float(new_k[new_offset]) : 0.0;
    const float new_v_value = valid_dim ? float(new_v[new_offset]) : 0.0;
    if (cache_writer && valid_dim && current_pos < pc.S) {
        const uint cache_offset = cache_head_base + current_pos * pc.D + dim;
        k_cache[cache_offset] = float16_t(new_k_value);
        v_cache[cache_offset] = float16_t(new_v_value);
    }

    const float q_value = valid_dim ? float(q[head * pc.D + dim]) : 0.0;
    const float scale = inversesqrt(float(pc.D));

    float running_max = NEG_INF;
    float running_sum = 0.0;
    float acc = 0.0;

    const uint cache_len = min(current_pos + 1u, pc.S);
    for (uint key_pos = 0u; key_pos < cache_len; ++key_pos) {
        float k_val = 0.0;
        float v_val = 0.0;
        if (key_pos == current_pos) {
            k_val = new_k_value;
            v_val = new_v_value;
        } else if (valid_dim) {
            const uint cache_offset = cache_head_base + key_pos * pc.D + dim;
            k_val = float(k_cache[cache_offset]);
            v_val = float(v_cache[cache_offset]);
        }

        const float dot_part = valid_dim ? q_value * k_val : 0.0;
        const float dot_sum = subgroupAdd(dot_part);
        if (gl_SubgroupInvocationID == 0u) {
            subgroup_dot[dim / gl_SubgroupSize] = dot_sum;
        }
        barrier();

        if (valid_dim) {
            float dot = 0.0;
            for (uint i = 0u; i < (pc.D + gl_SubgroupSize - 1u) / gl_SubgroupSize; ++i) {
                dot += subgroup_dot[i];
            }
            const float score = dot * scale;
            const float next_max = max(running_max, score);
            const float old_scale = running_max == NEG_INF ? 0.0 : exp(running_max - next_max);
            const float score_scale = exp(score - next_max);
            acc = acc * old_scale + score_scale * v_val;
            running_sum = running_sum * old_scale + score_scale;
            running_max = next_max;
        }
        barrier();
    }

    if (valid_dim && running_sum > 0.0) {
        output_values[head * pc.D + dim] = float16_t(acc / running_sum);
    }
}
""",
)
