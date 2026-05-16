"""Generated shader: sdpa_f16."""

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


SDPA_F16 = ShaderVariant(
    name='sdpa_f16',
    family='export',
    contract=ShaderContract(
        class_name='ExportSdpaProgram',
        shader_name='sdpa_f16',
        fields=(
            TensorFieldSpec(
                name='q',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('Q0', 'Q1', 'Q2', 'Q3',)),
            ),
            TensorFieldSpec(
                name='k',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('K0', 'K1', 'K2', 'K3',)),
            ),
            TensorFieldSpec(
                name='v',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('V0', 'V1', 'V2', 'V3',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=('O0', 'O1', 'O2', 'O3',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=24,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 'Q0', dynamic=False),
                PushConstantFieldSpec('NH', PushConstantType.UINT32, 4, 'Q1', dynamic=False),
                PushConstantFieldSpec('NK', PushConstantType.UINT32, 8, 'K1', dynamic=False),
                PushConstantFieldSpec('T', PushConstantType.UINT32, 12, 'Q2', dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 16, 'K2', dynamic=False),
                PushConstantFieldSpec('D', PushConstantType.UINT32, 20, 'Q3', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(mul('Q0', 'Q1'), 'Q2', 1),
    ),
    execution_requirements=ShaderExecutionRequirements(subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True), require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_subgroup_extended_types_float16 : require

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float16_t q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float16_t k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float16_t v[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint NH; uint NK; uint T; uint S; uint D; } pc;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
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
    const uint k_base = (batch * pc.NK + kv_head) * pc.S * pc.D;
    const uint v_base = k_base;
    const uint q_row_base = q_base + row * pc.D;
    const float16_t q0 = valid0 ? float16_t(q[q_row_base + dim0]) : float16_t(0.0);
    const float16_t q1 = valid1 ? float16_t(q[q_row_base + dim1]) : float16_t(0.0);
    const float16_t scale = float16_t(inversesqrt(float(pc.D)));
    float16_t running_max = float16_t(-65504.0);
    float16_t running_sum = float16_t(0.0);
    float16_t acc0 = float16_t(0.0);
    float16_t acc1 = float16_t(0.0);

    for (uint col = 0u; col < pc.S; ++col) {
        const uint kv_offset = col * pc.D;
        const float16_t k0 = valid0 ? float16_t(k[k_base + kv_offset + dim0]) : float16_t(0.0);
        const float16_t k1 = valid1 ? float16_t(k[k_base + kv_offset + dim1]) : float16_t(0.0);
        const float16_t dot = subgroupAdd(q0 * k0 + q1 * k1);
        const float16_t score = dot * scale;
        const float16_t next_max = score > running_max ? score : running_max;
        const float16_t old_scale = running_max == float16_t(-65504.0)
            ? float16_t(0.0)
            : float16_t(exp(float(running_max - next_max)));
        const float16_t score_scale = float16_t(exp(float(score - next_max)));
        if (valid0) {
            acc0 = acc0 * old_scale + score_scale * float16_t(v[v_base + kv_offset + dim0]);
        }
        if (valid1) {
            acc1 = acc1 * old_scale + score_scale * float16_t(v[v_base + kv_offset + dim1]);
        }
        running_sum = running_sum * old_scale + score_scale;
        running_max = next_max;
    }

    if (running_sum > float16_t(0.0)) {
        const uint output_base = (batch * pc.NH + head) * pc.T * pc.D + row * pc.D;
        if (valid0) {
            output_values[output_base + dim0] = float16_t(acc0 / running_sum);
        }
        if (valid1) {
            output_values[output_base + dim1] = float16_t(acc1 / running_sum);
        }
    }
}
""",
)
