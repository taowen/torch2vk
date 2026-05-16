"""Generated shader: sdpa_wide_f16."""

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


SDPA_WIDE_F16 = ShaderVariant(
    name='sdpa_wide_f16',
    family='export',
    contract=ShaderContract(
        class_name='ExportSdpaProgram',
        shader_name='sdpa_wide_f16',
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

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float16_t q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float16_t k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float16_t v[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint NH; uint NK; uint T; uint S; uint D; } pc;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const float NEG_INF = -3.4028234663852886e38;

void main() {
    const uint batch_head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    if (batch_head >= pc.B * pc.NH || row >= pc.T || pc.D > 512u) { return; }

    const uint batch = batch_head / pc.NH;
    const uint head = batch_head % pc.NH;
    const uint kv_head = head * pc.NK / pc.NH;
    const uint q_base = (batch * pc.NH + head) * pc.T * pc.D;
    const uint k_base = (batch * pc.NK + kv_head) * pc.S * pc.D;
    const uint v_base = k_base;
    const uint q_row_base = q_base + row * pc.D;
    const float scale = inversesqrt(float(pc.D));

    float running_max = NEG_INF;
    float running_sum = 0.0;
    float acc0 = 0.0;
    float acc1 = 0.0;
    float acc2 = 0.0;
    float acc3 = 0.0;
    float acc4 = 0.0;
    float acc5 = 0.0;
    float acc6 = 0.0;
    float acc7 = 0.0;

    for (uint col = 0u; col < pc.S; ++col) {
        const uint kv_offset = col * pc.D;
        float partial = 0.0;
        for (uint d = lane; d < pc.D; d += 64u) {
            partial += float(q[q_row_base + d]) * float(k[k_base + kv_offset + d]);
        }
        const float score = subgroupAdd(partial) * scale;
        const float next_max = max(running_max, score);
        const float old_scale = running_max == NEG_INF ? 0.0 : exp(running_max - next_max);
        const float score_scale = exp(score - next_max);
        if (lane < pc.D) { acc0 = acc0 * old_scale + score_scale * float(v[v_base + kv_offset + lane]); }
        if (lane + 64u < pc.D) { acc1 = acc1 * old_scale + score_scale * float(v[v_base + kv_offset + lane + 64u]); }
        if (lane + 128u < pc.D) { acc2 = acc2 * old_scale + score_scale * float(v[v_base + kv_offset + lane + 128u]); }
        if (lane + 192u < pc.D) { acc3 = acc3 * old_scale + score_scale * float(v[v_base + kv_offset + lane + 192u]); }
        if (lane + 256u < pc.D) { acc4 = acc4 * old_scale + score_scale * float(v[v_base + kv_offset + lane + 256u]); }
        if (lane + 320u < pc.D) { acc5 = acc5 * old_scale + score_scale * float(v[v_base + kv_offset + lane + 320u]); }
        if (lane + 384u < pc.D) { acc6 = acc6 * old_scale + score_scale * float(v[v_base + kv_offset + lane + 384u]); }
        if (lane + 448u < pc.D) { acc7 = acc7 * old_scale + score_scale * float(v[v_base + kv_offset + lane + 448u]); }
        running_sum = running_sum * old_scale + score_scale;
        running_max = next_max;
    }

    if (running_sum > 0.0) {
        const uint output_base = (batch * pc.NH + head) * pc.T * pc.D + row * pc.D;
        if (lane < pc.D) { output_values[output_base + lane] = float16_t(acc0 / running_sum); }
        if (lane + 64u < pc.D) { output_values[output_base + lane + 64u] = float16_t(acc1 / running_sum); }
        if (lane + 128u < pc.D) { output_values[output_base + lane + 128u] = float16_t(acc2 / running_sum); }
        if (lane + 192u < pc.D) { output_values[output_base + lane + 192u] = float16_t(acc3 / running_sum); }
        if (lane + 256u < pc.D) { output_values[output_base + lane + 256u] = float16_t(acc4 / running_sum); }
        if (lane + 320u < pc.D) { output_values[output_base + lane + 320u] = float16_t(acc5 / running_sum); }
        if (lane + 384u < pc.D) { output_values[output_base + lane + 384u] = float16_t(acc6 / running_sum); }
        if (lane + 448u < pc.D) { output_values[output_base + lane + 448u] = float16_t(acc7 / running_sum); }
    }
}
""",
)
