"""Prefill causal SDPA with float32 Q/O and float16 K/V cache."""

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


SDPA_CAUSAL_CACHE_WRITE_MIXED_F32 = ShaderVariant(
    name="sdpa_causal_cache_write_mixed_f32",
    family="optimized_qwen3",
    contract=ShaderContract(
        class_name="OptimizedQwen3SdpaCausalCacheWriteMixedF32Program",
        shader_name="sdpa_causal_cache_write_mixed_f32",
        fields=(
            TensorFieldSpec("q", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("B", "NH", "T", "D"))),
            TensorFieldSpec("k", IOKind.INPUT, "input", TensorContract(dtype="float16", shape=("B", "NK", "T", "D"))),
            TensorFieldSpec("v", IOKind.INPUT, "input", TensorContract(dtype="float16", shape=("B", "T", "NK", "D"))),
            TensorFieldSpec("k_cache", IOKind.INOUT, "state", TensorContract(dtype="float16", shape=("B", "NK", "S", "D"))),
            TensorFieldSpec("v_cache", IOKind.INOUT, "state", TensorContract(dtype="float16", shape=("B", "NK", "S", "D"))),
            TensorFieldSpec("cache_position", IOKind.INPUT, "cache_position", TensorContract(dtype="int64", shape=("T",))),
            TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=("B", "NH", "T", "D"))),
        ),
        push_constants=PushConstantSpec(
            size=24,
            fields=(
                PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                PushConstantFieldSpec("NH", PushConstantType.UINT32, 4, "NH"),
                PushConstantFieldSpec("NK", PushConstantType.UINT32, 8, "NK"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 12, "T"),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 16, "S"),
                PushConstantFieldSpec("D", PushConstantType.UINT32, 20, "D"),
            ),
        ),
        dispatch=(mul("B", "NH"), "T", 1),
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
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float16_t k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float16_t v[]; };
layout(set = 0, binding = 3) buffer restrict KCacheBuffer { float16_t k_cache[]; };
layout(set = 0, binding = 4) buffer restrict VCacheBuffer { float16_t v_cache[]; };
layout(set = 0, binding = 5) buffer restrict readonly CachePositionBuffer { int64_t cache_position[]; };
layout(set = 0, binding = 6) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint NH; uint NK; uint T; uint S; uint D; } pc;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
const float NEG_INF = -3.4028234663852886e38;
void main() {
    const uint batch_head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint dim0 = gl_LocalInvocationID.x;
    const uint dim1 = dim0 + 64u;
    if (batch_head >= pc.B * pc.NH || row >= pc.T) { return; }
    const bool valid0 = dim0 < pc.D;
    const bool valid1 = dim1 < pc.D;
    const uint batch = batch_head / pc.NH;
    const uint head = batch_head % pc.NH;
    const uint kv_head = head * pc.NK / pc.NH;
    const uint q_base = (batch * pc.NH + head) * pc.T * pc.D;
    const uint k_base = (batch * pc.NK + kv_head) * pc.T * pc.D;
    const uint cache_base = (batch * pc.NK + kv_head) * pc.S * pc.D;
    const uint q_row_base = q_base + row * pc.D;
    const float q0 = valid0 ? q[q_row_base + dim0] : 0.0;
    const float q1 = valid1 ? q[q_row_base + dim1] : 0.0;
    const float scale = inversesqrt(float(pc.D));
    const uint q_heads_per_kv = pc.NH / pc.NK;
    const bool cache_writer = head == kv_head * q_heads_per_kv;
    const uint dst_t = uint(cache_position[row]);
    if (cache_writer && dst_t < pc.S) {
        const uint k_row_base = k_base + row * pc.D;
        const uint cache_row_base = cache_base + dst_t * pc.D;
        if (valid0) {
            k_cache[cache_row_base + dim0] = k[k_row_base + dim0];
            v_cache[cache_row_base + dim0] = v[((batch * pc.T + row) * pc.NK + kv_head) * pc.D + dim0];
        }
        if (valid1) {
            k_cache[cache_row_base + dim1] = k[k_row_base + dim1];
            v_cache[cache_row_base + dim1] = v[((batch * pc.T + row) * pc.NK + kv_head) * pc.D + dim1];
        }
    }
    float running_max = NEG_INF;
    float running_sum = 0.0;
    float acc0 = 0.0;
    float acc1 = 0.0;
    for (uint col = 0u; col <= row && col < pc.S; ++col) {
        const uint kv_offset = col * pc.D;
        const float k0 = valid0 ? float(k[k_base + kv_offset + dim0]) : 0.0;
        const float k1 = valid1 ? float(k[k_base + kv_offset + dim1]) : 0.0;
        const uint v_base = ((batch * pc.T + col) * pc.NK + kv_head) * pc.D;
        const float dot = subgroupAdd(q0 * k0 + q1 * k1);
        const float score = dot * scale;
        const float next_max = max(running_max, score);
        const float old_scale = running_max == NEG_INF ? 0.0 : exp(running_max - next_max);
        const float score_scale = exp(score - next_max);
        if (valid0) { acc0 = acc0 * old_scale + score_scale * float(v[v_base + dim0]); }
        if (valid1) { acc1 = acc1 * old_scale + score_scale * float(v[v_base + dim1]); }
        running_sum = running_sum * old_scale + score_scale;
        running_max = next_max;
    }
    if (running_sum > 0.0) {
        const uint output_base = (batch * pc.NH + head) * pc.T * pc.D + row * pc.D;
        if (valid0) { output_values[output_base + dim0] = acc0 / running_sum; }
        if (valid1) { output_values[output_base + dim1] = acc1 / running_sum; }
    }
}
""",
)
