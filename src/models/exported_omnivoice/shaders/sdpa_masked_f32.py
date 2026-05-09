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
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

const float NEG_INF = -3.4028234663852886e38;

shared float subgroup_dot[2];

void main() {
    const uint batch_head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint dim = gl_LocalInvocationID.x;
    if (batch_head >= pc.B * pc.NH || row >= pc.T) { return; }
    const bool valid_dim = dim < pc.D;

    const uint batch = batch_head / pc.NH;
    const uint head = batch_head % pc.NH;
    const uint kv_head = head * pc.NK / pc.NH;
    const uint q_base = (batch * pc.NH + head) * pc.T * pc.D;
    const uint k_base = (batch * pc.NK + kv_head) * pc.S * pc.D;
    const uint v_base = k_base;
    const uint mask_base = batch * pc.T * pc.S;
    const float q_value = valid_dim ? q[q_base + row * pc.D + dim] : 0.0;
    const float scale = inversesqrt(float(pc.D));

    float running_max = NEG_INF;
    float running_sum = 0.0;
    float acc = 0.0;

    for (uint col = 0u; col < pc.S; ++col) {
        const float k_value = valid_dim ? k[k_base + col * pc.D + dim] : 0.0;
        const float v_value = valid_dim ? v[v_base + col * pc.D + dim] : 0.0;
        const float dot_part = valid_dim ? q_value * k_value : 0.0;
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
            const float score = dot * scale + mask[mask_base + row * pc.S + col];
            const float next_max = max(running_max, score);
            const float old_scale = running_max == NEG_INF ? 0.0 : exp(running_max - next_max);
            const float score_scale = exp(score - next_max);
            acc = acc * old_scale + score_scale * v_value;
            running_sum = running_sum * old_scale + score_scale;
            running_max = next_max;
        }
        barrier();
    }

    if (valid_dim && running_sum > 0.0) {
        output_values[(batch * pc.NH + head) * pc.T * pc.D + row * pc.D + dim] = acc / running_sum;
    }
}
""",
)
