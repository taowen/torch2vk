from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_store,
    make_unary_elementwise,
)
from torch2vk.runtime.shader import ShaderVariant

_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
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
        output_values[idx] = {{STORE_GELU}};
    }
}
"""


def make_gelu_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    return make_unary_elementwise(
        _source(activation_dtype),
        "gelu_f32",
        node,
        input_dtype=activation_dtype,
        output_dtype=activation_dtype,
    )


def _source(activation_dtype: str) -> str:
    return (
        _SOURCE
        .replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_GELU}}", activation_store("gelu_erf(float(x[idx]))", activation_dtype))
    )
