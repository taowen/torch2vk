from __future__ import annotations

from torch.fx import Node

from torch2vk.export.dtype_policy import requires_float32_intermediate
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
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{OUTPUT_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { float v = float(x[idx]); output_values[idx] = {{STORE_POW}}; }
}
"""


def make_pow_scalar_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    output_dtype = "float32" if requires_float32_intermediate(node) else activation_dtype
    shader_name = "pow_scalar_f32" if output_dtype == "float32" else "pow_scalar_f16"
    return make_unary_elementwise(
        _source(activation_dtype=activation_dtype, output_dtype=output_dtype),
        shader_name,
        node,
        input_dtype=activation_dtype,
        output_dtype=output_dtype,
    )


def _source(*, activation_dtype: str, output_dtype: str) -> str:
    return (
        _SOURCE.replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{OUTPUT_TYPE}}", activation_glsl_type(output_dtype))
        .replace("{{STORE_POW}}", activation_store("v * v", output_dtype))
    )
