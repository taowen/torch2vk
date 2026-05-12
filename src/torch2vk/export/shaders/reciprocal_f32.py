from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    make_unary_elementwise,
    node_input_dtype,
)
from torch2vk.runtime.shader import ShaderVariant

_SOURCE = """\
#version 450
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = 1.0 / float(x[idx]); }
}
"""


def make_reciprocal_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    input_dtype = node_input_dtype(node, 0) or activation_dtype
    return make_unary_elementwise(
        _source(input_dtype),
        "reciprocal_f32",
        node,
        input_dtype=input_dtype,
        output_dtype="float32",
    )


def _source(activation_dtype: str) -> str:
    return _SOURCE.replace(
        "{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype)
    ).replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
