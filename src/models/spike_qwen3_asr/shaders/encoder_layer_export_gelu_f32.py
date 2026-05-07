"""Generated shader: encoder_layer_export_gelu_f32."""

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
    ceil_div,
    mul,
)


ENCODER_LAYER_EXPORT_GELU_F32 = ShaderVariant(
    name='encoder_layer_export_gelu_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportGeluF32Program',
        shader_name='encoder_layer_export_gelu_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('B', 'T',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('B', 'T',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul('B', 'T'), dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul('B', 'T'), 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
float erf_approx(float value) {
    const float p = 0.3275911;
    const float a1 = 0.254829592;
    const float a2 = -0.284496736;
    const float a3 = 1.421413741;
    const float a4 = -1.453152027;
    const float a5 = 1.061405429;
    const float sign_value = value < 0.0 ? -1.0 : 1.0;
    const float abs_value = abs(value);
    const float t = 1.0 / (1.0 + p * abs_value);
    const float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-abs_value * abs_value);
    return sign_value * y;
}
float gelu_erf(float value) {
    return 0.5 * value * (1.0 + erf_approx(value * 0.7071067811865476));
}
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        output_values[idx] = gelu_erf(x[idx]);
    }
}
""",
)
