from __future__ import annotations

import math

from torch.fx import Node

from torch2vk.export.shaders._factory import (
    activation_extension_source,
    activation_glsl_type,
    activation_requirements,
    activation_store,
    node_input_shape,
)
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
{{ACTIVATION_EXTENSION}}\
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { {{ACTIVATION_TYPE}} x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { {{ACTIVATION_TYPE}} output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float partial_max[256];
void main() {
    const uint tid = gl_LocalInvocationID.x;
    float local_max = -1.0e38;
    for (uint i = tid; i < pc.N; i += 256u) {
        local_max = max(local_max, float(x[i]));
    }
    partial_max[tid] = local_max;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial_max[tid] = max(partial_max[tid], partial_max[tid + stride]); }
        barrier();
    }
    if (tid == 0u) { output_values[0] = {{STORE_MAX}}; }
}
"""


def make_max_variant(node: Node, activation_dtype: str = "float32") -> ShaderVariant | None:
    in_shape = node_input_shape(node, 0)
    if not in_shape:
        return None

    n = math.prod(in_shape)
    in_contract = tuple(f"I{i}" for i in range(len(in_shape)))

    return ShaderVariant(
        name="max_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportMaxProgram",
            shader_name="max_f32",
            fields=(
                TensorFieldSpec(
                    "x",
                    IOKind.INPUT,
                    "input",
                    TensorContract(dtype=activation_dtype, shape=in_contract),
                ),
                TensorFieldSpec(
                    "output",
                    IOKind.OUTPUT,
                    "output",
                    TensorContract(dtype=activation_dtype, shape=(1,)),
                ),
            ),
            push_constants=PushConstantSpec(
                size=4,
                fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, n),),
            ),
            dispatch=(1, 1, 1),
        ),
        source=_source(activation_dtype),
        execution_requirements=activation_requirements(activation_dtype),
    )


def _source(activation_dtype: str) -> str:
    return (
        _SOURCE.replace("{{ACTIVATION_EXTENSION}}", activation_extension_source(activation_dtype))
        .replace("{{ACTIVATION_TYPE}}", activation_glsl_type(activation_dtype))
        .replace("{{STORE_MAX}}", activation_store("partial_max[0]", activation_dtype))
    )
