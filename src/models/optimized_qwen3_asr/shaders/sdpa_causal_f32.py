"""Generated shader: sdpa_causal_f32."""

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


SDPA_CAUSAL_F32 = ShaderVariant(
    name='sdpa_causal_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportSdpaProgram',
        shader_name='sdpa_causal_f32',
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
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O0', 'O1', 'O2', 'O3',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=24,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 1, dynamic=False),
                PushConstantFieldSpec('NH', PushConstantType.UINT32, 4, 16, dynamic=False),
                PushConstantFieldSpec('NK', PushConstantType.UINT32, 8, 8, dynamic=False),
                PushConstantFieldSpec('T', PushConstantType.UINT32, 12, 151, dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 16, 151, dynamic=False),
                PushConstantFieldSpec('D', PushConstantType.UINT32, 20, 128, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(16, 151, 2),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float v[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint NH; uint NK; uint T; uint S; uint D; } pc;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint batch_head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint d_out = gl_WorkGroupID.z * 64u + gl_LocalInvocationID.x;
    if (batch_head >= pc.B * pc.NH || row >= pc.T || d_out >= pc.D) { return; }
    const uint batch = batch_head / pc.NH;
    const uint head = batch_head % pc.NH;
    const uint kv_head = head * pc.NK / pc.NH;
    const uint q_base = (batch * pc.NH + head) * pc.T * pc.D;
    const uint k_base = (batch * pc.NK + kv_head) * pc.S * pc.D;
    const uint v_base = k_base;
    const float scale = inversesqrt(float(pc.D));
    float max_score = -1.0e38;
    for (uint col = 0u; col <= row && col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[q_base + row * pc.D + d] * k[k_base + col * pc.D + d];
        }
        max_score = max(max_score, dot * scale);
    }
    float sum_exp = 0.0;
    for (uint col = 0u; col <= row && col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[q_base + row * pc.D + d] * k[k_base + col * pc.D + d];
        }
        sum_exp += exp(dot * scale - max_score);
    }
    float acc = 0.0;
    for (uint col = 0u; col <= row && col < pc.S; ++col) {
        float dot = 0.0;
        for (uint dd = 0u; dd < pc.D; ++dd) {
            dot += q[q_base + row * pc.D + dd] * k[k_base + col * pc.D + dd];
        }
        float w = exp(dot * scale - max_score) / sum_exp;
        acc += w * v[v_base + col * pc.D + d_out];
    }
    output_values[(batch * pc.NH + head) * pc.T * pc.D + row * pc.D + d_out] = acc;
}
""",
)
