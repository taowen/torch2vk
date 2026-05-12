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
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = {{STORE_SIN}}; }
}
"""


def make_sin_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    return make_unary_elementwise(
        _source(activation_dtype),
        "sin_f32",
        node,
        input_dtype=activation_dtype,
        output_dtype=activation_dtype,
    )


def _source(activation_dtype: str) -> str:
    return (
        _SOURCE.replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_SIN}}", activation_store("sin(float(x[idx]))", activation_dtype))
    )
