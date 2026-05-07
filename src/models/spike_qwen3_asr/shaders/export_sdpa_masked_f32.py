"""Generated shader: export_sdpa_masked_f32."""

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


EXPORT_SDPA_MASKED_F32 = ShaderVariant(
    name='export_sdpa_masked_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportSdpaProgram',
        shader_name='export_sdpa_masked_f32',
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
            size=20,
            fields=(
                PushConstantFieldSpec('NH', PushConstantType.UINT32, 0, 14),
                PushConstantFieldSpec('NK', PushConstantType.UINT32, 4, 14),
                PushConstantFieldSpec('T', PushConstantType.UINT32, 8, 133),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 12, 133),
                PushConstantFieldSpec('D', PushConstantType.UINT32, 16, 64),
            ),
        ),
        dispatch=(14, 133, 1),
    ),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float v[]; };
layout(set = 0, binding = 3) buffer restrict readonly MaskBuffer { float mask[]; };
layout(set = 0, binding = 4) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint NH; uint NK; uint T; uint S; uint D; } pc;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    if (head >= pc.NH || row >= pc.T) { return; }
    const uint kv_head = head * pc.NK / pc.NH;
    const float scale = inversesqrt(float(pc.D));
    float max_score = -1.0e38;
    for (uint col = 0u; col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[(head * pc.T + row) * pc.D + d] * k[(kv_head * pc.S + col) * pc.D + d];
        }
        float s = dot * scale + mask[row * pc.S + col];
        max_score = max(max_score, s);
    }
    float sum_exp = 0.0;
    for (uint col = 0u; col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[(head * pc.T + row) * pc.D + d] * k[(kv_head * pc.S + col) * pc.D + d];
        }
        sum_exp += exp(dot * scale + mask[row * pc.S + col] - max_score);
    }
    for (uint d = 0u; d < pc.D; ++d) {
        float acc = 0.0;
        for (uint col = 0u; col < pc.S; ++col) {
            float dot = 0.0;
            for (uint dd = 0u; dd < pc.D; ++dd) {
                dot += q[(head * pc.T + row) * pc.D + dd] * k[(kv_head * pc.S + col) * pc.D + dd];
            }
            float w = exp(dot * scale + mask[row * pc.S + col] - max_score) / sum_exp;
            acc += w * v[(kv_head * pc.S + col) * pc.D + d];
        }
        output_values[(head * pc.T + row) * pc.D + d] = acc;
    }
}
""",
)
