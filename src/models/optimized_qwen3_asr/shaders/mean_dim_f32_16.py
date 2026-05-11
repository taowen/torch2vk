"""Generated shader: mean_dim_f32_16."""

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
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
    SubgroupRequirements,
)


MEAN_DIM_F32_16 = ShaderVariant(
    name='mean_dim_f32_16',
    family='export',
    contract=ShaderContract(
        class_name='ExportMeanDimProgram',
        shader_name='mean_dim_f32_16',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('S0', 'S1', 'S2', 'S3',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('O0', 'O1', 'O2', 'O3',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('ROWS', PushConstantType.UINT32, 0, 1208, dynamic=False),
                PushConstantFieldSpec('COLS', PushConstantType.UINT32, 4, 128, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(1208, 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True)),
    source="""\
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
""",
)
