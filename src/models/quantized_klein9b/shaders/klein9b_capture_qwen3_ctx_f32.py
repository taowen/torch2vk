"""Generated shader: klein9b_capture_qwen3_ctx_f32."""

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


KLEIN9B_CAPTURE_QWEN3_CTX_F32 = ShaderVariant(
    name='klein9b_capture_qwen3_ctx_f32',
    family='klein9b',
    contract=ShaderContract(
        class_name='Klein9BCaptureQwen3CtxF32Program',
        shader_name='klein9b_capture_qwen3_ctx_f32',
        fields=(
            TensorFieldSpec(
                name='layer_9',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('B', 'S', 'H',)),
            ),
            TensorFieldSpec(
                name='layer_18',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('B', 'S', 'H',)),
            ),
            TensorFieldSpec(
                name='layer_27',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('B', 'S', 'H',)),
            ),
            TensorFieldSpec(
                name='ctx',
                io_kind=IOKind.OUTPUT,
                role='output',
                contract=TensorContract(dtype='float32', shape=('B', 'S', mul(3, 'H'),)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec('S', PushConstantType.UINT32, 0, 'S', dynamic=False),
                PushConstantFieldSpec('H', PushConstantType.UINT32, 4, 'H', dynamic=False),
                PushConstantFieldSpec('N', PushConstantType.UINT32, 8, mul(mul('B', 'S'), 'H'), dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('B', 'S'), 'H'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly Layer9Buffer { float16_t layer_9[]; };
layout(set = 0, binding = 1) buffer restrict readonly Layer18Buffer { float16_t layer_18[]; };
layout(set = 0, binding = 2) buffer restrict readonly Layer27Buffer { float16_t layer_27[]; };
layout(set = 0, binding = 3) buffer restrict writeonly CtxBuffer { float ctx[]; };
layout(push_constant) uniform PushConstants { uint S; uint H; uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const uint hidden = idx % pc.H;
    const uint token = (idx / pc.H) % pc.S;
    const uint batch = idx / (pc.H * pc.S);
    const uint out_base = (batch * pc.S + token) * (3u * pc.H) + hidden;
    ctx[out_base] = float(layer_9[idx]);
    ctx[out_base + pc.H] = float(layer_18[idx]);
    ctx[out_base + 2u * pc.H] = float(layer_27[idx]);
}
""",
)
