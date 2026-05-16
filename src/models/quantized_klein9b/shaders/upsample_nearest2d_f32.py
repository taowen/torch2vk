"""Generated shader: upsample_nearest2d_f32."""

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
    ceil_div,
    mul,
)
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


UPSAMPLE_NEAREST2D_F32 = ShaderVariant(
    name='upsample_nearest2d_f32',
    family='export',
    contract=ShaderContract(
        class_name='ExportUpsampleNearest2dProgram',
        shader_name='upsample_nearest2d_f32',
        fields=(
            TensorFieldSpec(
                name='x',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=(1, 'C', 'H', 'W',)),
            ),
            TensorFieldSpec(
                name='output',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float16', shape=(1, 'C', 'OH', 'OW',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=24,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 1, dynamic=False),
                PushConstantFieldSpec('C', PushConstantType.UINT32, 4, 'C', dynamic=False),
                PushConstantFieldSpec('H', PushConstantType.UINT32, 8, 'H', dynamic=False),
                PushConstantFieldSpec('W', PushConstantType.UINT32, 12, 'W', dynamic=False),
                PushConstantFieldSpec('OH', PushConstantType.UINT32, 16, 'OH', dynamic=False),
                PushConstantFieldSpec('OW', PushConstantType.UINT32, 20, 'OW', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul(mul(1, 'C'), 'OH'), 'OW'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint C; uint H; uint W; uint OH; uint OW; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.C * pc.OH * pc.OW;
    if (idx >= total) { return; }
    const uint ow = idx % pc.OW;
    const uint oh = (idx / pc.OW) % pc.OH;
    const uint c = (idx / (pc.OW * pc.OH)) % pc.C;
    const uint b = idx / (pc.OW * pc.OH * pc.C);
    const uint ih = min(oh * pc.H / pc.OH, pc.H - 1u);
    const uint iw = min(ow * pc.W / pc.OW, pc.W - 1u);
    const uint input_idx = ((b * pc.C + c) * pc.H + ih) * pc.W + iw;
    output_values[idx] = float16_t(float(x[input_idx]));
}
""",
)
