"""Generated shader: index_copy_f32_7ba4f1ff13."""

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


INDEX_COPY_F32_7BA4F1FF13 = ShaderVariant(
    name='index_copy_f32_7ba4f1ff13',
    family='export',
    contract=ShaderContract(
        class_name='ExportIndexCopyF32Program',
        shader_name='index_copy_f32_7ba4f1ff13',
        fields=(
            TensorFieldSpec(
                name='cache',
                io_kind=IOKind.INOUT,
                role='state',
                contract=TensorContract(dtype='float16', shape=('B', 'T', 'H',)),
            ),
            TensorFieldSpec(
                name='index',
                io_kind=IOKind.INPUT,
                role='index',
                contract=TensorContract(dtype='int64', shape=('S',)),
            ),
            TensorFieldSpec(
                name='src',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float16', shape=('B', 'S', 'H',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('T', PushConstantType.UINT32, 4, 'T', dynamic=False),
                PushConstantFieldSpec('H', PushConstantType.UINT32, 8, 'H', dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 12, 'S', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('B', 'S'), 'H'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True, require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict CacheBuffer { float16_t cache[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndexBuffer { int64_t index_values[]; };
layout(set = 0, binding = 2) buffer restrict readonly SrcBuffer { float16_t src[]; };
layout(push_constant) uniform PushConstants { uint B; uint T; uint H; uint S; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.S * pc.H;
    if (idx >= total) { return; }

    uint rem = idx;
    const uint h = rem % pc.H;
    rem = rem / pc.H;
    const uint s = rem % pc.S;
    const uint b = rem / pc.S;

    const uint dst_t = uint(index_values[s]);
    const uint src_idx = (b * pc.S + s) * pc.H + h;
    const uint dst_idx = (b * pc.T + dst_t) * pc.H + h;
    cache[dst_idx] = src[src_idx];
}
""",
)
