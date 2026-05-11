from __future__ import annotations

from torch.fx import Node

from torch2vk.export.shaders._factory import make_unary_elementwise
from torch2vk.runtime.shader import ShaderVariant

_SOURCE = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = inversesqrt(float(x[idx])); }
}
"""


def make_rsqrt_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    return make_unary_elementwise(_SOURCE, "rsqrt_f32", node)
