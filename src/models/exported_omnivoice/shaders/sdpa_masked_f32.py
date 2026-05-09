"""Generated shader: sdpa_masked_f32."""

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


SDPA_MASKED_F32 = ShaderVariant(
    name='sdpa_masked_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportSdpaProgram',
        shader_name='sdpa_masked_f32',
        fields=(
            TensorFieldSpec(
                name='q',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('Q0', 'Q1', 'Q2', 'Q3',)),
            ),
            TensorFieldSpec(
                name='k',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('K0', 'K1', 'K2', 'K3',)),
            ),
            TensorFieldSpec(
                name='v',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('V0', 'V1', 'V2', 'V3',)),
            ),
            TensorFieldSpec(
                name='mask',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('M0', 'M1', 'M2', 'M3',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O0', 'O1', 'O2', 'O3',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=24,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 2, dynamic=False),
                PushConstantFieldSpec('NH', PushConstantType.UINT32, 4, 16, dynamic=False),
                PushConstantFieldSpec('NK', PushConstantType.UINT32, 8, 8, dynamic=False),
                PushConstantFieldSpec('T', PushConstantType.UINT32, 12, 300, dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 16, 300, dynamic=False),
                PushConstantFieldSpec('D', PushConstantType.UINT32, 20, 128, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(32, 300, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True)),
    source="""\
#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float v[]; };
layout(set = 0, binding = 3) buffer restrict readonly MaskBuffer { float mask[]; };
layout(set = 0, binding = 4) buffer restrict writeonly OutputBuffer { float output_values[]; };
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
    const uint k_base = (batch * pc.NK + kv_head) * pc.S * pc.D;
    const uint v_base = k_base;
    const uint mask_base = batch * pc.T * pc.S;
    const uint q_row_base = q_base + row * pc.D;
    const float q0 = valid0 ? q[q_row_base + dim0] : 0.0;
    const float q1 = valid1 ? q[q_row_base + dim1] : 0.0;
    const float scale = inversesqrt(float(pc.D));

    float running_max = NEG_INF;
    float running_sum = 0.0;
    float acc0 = 0.0;
    float acc1 = 0.0;

    for (uint col = 0u; col < pc.S; ++col) {
        const uint kv_offset = col * pc.D;
        const float k0 = valid0 ? k[k_base + kv_offset + dim0] : 0.0;
        const float k1 = valid1 ? k[k_base + kv_offset + dim1] : 0.0;
        const float dot = subgroupAdd(q0 * k0 + q1 * k1);
        const float score = dot * scale + mask[mask_base + row * pc.S + col];
        const float next_max = max(running_max, score);
        const float old_scale = running_max == NEG_INF ? 0.0 : exp(running_max - next_max);
        const float score_scale = exp(score - next_max);
        if (valid0) {
            acc0 = acc0 * old_scale + score_scale * v[v_base + kv_offset + dim0];
        }
        if (valid1) {
            acc1 = acc1 * old_scale + score_scale * v[v_base + kv_offset + dim1];
        }
        running_sum = running_sum * old_scale + score_scale;
        running_max = next_max;
    }

    if (running_sum > 0.0) {
        const uint output_base = (batch * pc.NH + head) * pc.T * pc.D + row * pc.D;
        if (valid0) {
            output_values[output_base + dim0] = acc0 / running_sum;
        }
        if (valid1) {
            output_values[output_base + dim1] = acc1 / running_sum;
        }
    }
}
""",
)
