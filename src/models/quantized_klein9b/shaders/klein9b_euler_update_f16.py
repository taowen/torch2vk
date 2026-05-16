"""Generated shader: klein9b_euler_update_f16."""

from __future__ import annotations

from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantInput,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
    ceil_div,
    mul,
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


KLEIN9B_EULER_UPDATE_F16 = ShaderVariant(
    name='klein9b_euler_update_f16',
    family='klein9b',
    contract=ShaderContract(
        class_name='Klein9BEulerUpdateF16Program',
        shader_name='klein9b_euler_update_f16',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INOUT,
                role='state',
                contract=TensorContract(dtype='float16', shape=('B', 'S', 'C',)),
            ),
            TensorFieldSpec(
                name='pred',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('B', 'S', 'C',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec('N', PushConstantType.UINT32, 0, mul(mul('B', 'S'), 'C'), dynamic=False),
                PushConstantFieldSpec('dt', PushConstantType.FLOAT32, 4, PushConstantInput('dt'), dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('B', 'S'), 'C'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict readonly PredBuffer { float16_t pred[]; };
layout(push_constant) uniform PushConstants { uint N; float dt; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    x[idx] = float16_t(float(x[idx]) + pc.dt * float(pred[idx]));
}
""",
)
