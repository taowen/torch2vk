from __future__ import annotations

import math

from torch.fx import Node

from torch2vk.export.shaders._factory import node_input_shape, node_output_shape
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

_SOURCE = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float partial_max[256];
void main() {
    const uint tid = gl_LocalInvocationID.x;
    float local_max = -1.0e38;
    for (uint i = tid; i < pc.N; i += 256u) {
        local_max = max(local_max, x[i]);
    }
    partial_max[tid] = local_max;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial_max[tid] = max(partial_max[tid], partial_max[tid + stride]); }
        barrier();
    }
    if (tid == 0u) { output_values[0] = partial_max[0]; }
}
"""


def make_max_variant(node: Node) -> ShaderVariant | None:
    in_shape = node_input_shape(node, 0)
    if not in_shape:
        return None

    n = math.prod(in_shape)
    in_contract = tuple(f"I{i}" for i in range(len(in_shape)))

    return ShaderVariant(
        name="export_max_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportMaxProgram",
            shader_name="export_max_f32",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=in_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=(1,))),
            ),
            push_constants=PushConstantSpec(
                size=4,
                fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n),),
            ),
            dispatch=(1, 1, 1),
        ),
        source=_SOURCE,
    )
