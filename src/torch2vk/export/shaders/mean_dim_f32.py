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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements, SubgroupRequirements

_SOURCE = """\
#version 450
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint ROWS; uint COLS; } pc;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint lane = gl_SubgroupInvocationID;
    if (row >= pc.ROWS) { return; }
    float sum = 0.0;
    for (uint c = lane; c < pc.COLS; c += 64u) {
        sum += x[row * pc.COLS + c];
    }
    sum = subgroupAdd(sum);
    if (lane == 0u) { output_values[row] = sum / float(pc.COLS); }
}
"""


def make_mean_dim_variant(node: Node) -> ShaderVariant | None:
    in_shape = node_input_shape(node, 0)
    out_shape = node_output_shape(node)
    if not in_shape or not out_shape:
        return None

    rows = math.prod(out_shape)
    cols = math.prod(in_shape) // rows if rows > 0 else 1

    in_contract = tuple("S" + str(i) for i in range(len(in_shape)))
    out_contract = tuple("O" + str(i) for i in range(len(out_shape)))

    return ShaderVariant(
        name="mean_dim_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportMeanDimProgram",
            shader_name="mean_dim_f32",
            fields=(
                TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=in_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
            push_constants=PushConstantSpec(
                size=8,
                fields=(
                    PushConstantFieldSpec("ROWS", PushConstantType.UINT32, 0, rows),
                    PushConstantFieldSpec("COLS", PushConstantType.UINT32, 4, cols),
                ),
            ),
            dispatch=(rows, 1, 1),
        ),
        source=_SOURCE,
        execution_requirements=ShaderExecutionRequirements(
            subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        ),
    )
