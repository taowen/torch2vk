"""Generated shader: sdpa_decode_cache_f16."""

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


SDPA_DECODE_CACHE_F16 = ShaderVariant(
    name='sdpa_decode_cache_f16',
    family='export',
    contract=ShaderContract(
        class_name='ExportSdpaDecodeCacheF16Program',
        shader_name='sdpa_decode_cache_f16',
        fields=(
            TensorFieldSpec(
                name='q',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('B', 'NH', 'T', 'D',)),
            ),
            TensorFieldSpec(
                name='k',
                io_kind=IOKind.INPUT,
                role='state',
                contract=TensorContract(dtype='float16', shape=('B', 'NK', 'S', 'D',)),
            ),
            TensorFieldSpec(
                name='v',
                io_kind=IOKind.INPUT,
                role='state',
                contract=TensorContract(dtype='float16', shape=('B', 'NK', 'S', 'D',)),
            ),
            TensorFieldSpec(
                name='cache_position',
                io_kind=IOKind.INPUT,
                role='cache_position',
                contract=TensorContract(dtype='int64', shape=('T',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('B', 'NH', 'T', 'D',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec('NH', PushConstantType.UINT32, 0, 'NH', dynamic=False),
                PushConstantFieldSpec('NK', PushConstantType.UINT32, 4, 'NK', dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 8, 'S', dynamic=False),
                PushConstantFieldSpec('D', PushConstantType.UINT32, 12, 'D', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=('NH', 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True), require_shader_int64=True, require_storage_buffer_16bit_access=True),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float16_t q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float16_t k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float16_t v[]; };
layout(set = 0, binding = 3) buffer restrict readonly CachePositionBuffer { int64_t cache_position[]; };
layout(set = 0, binding = 4) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
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
    const float q_value = valid_dim ? float(q[head * pc.D + dim]) : 0.0;
    const float scale = inversesqrt(float(pc.D));

    float running_max = NEG_INF;
    float running_sum = 0.0;
    float acc = 0.0;

    const uint cache_head_base = kv_head * pc.S * pc.D;
    const uint cache_len = min(uint(cache_position[0]) + 1u, pc.S);
    for (uint key_pos = 0u; key_pos < cache_len; ++key_pos) {
        const float k_val = valid_dim ? float(k[cache_head_base + key_pos * pc.D + dim]) : 0.0;
        const float v_val = valid_dim ? float(v[cache_head_base + key_pos * pc.D + dim]) : 0.0;

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
