"""Generated shader: export_index_copy_f32_aced7cb1d3."""

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


EXPORT_INDEX_COPY_F32_ACED7CB1D3 = ShaderVariant(
    name='export_index_copy_f32_aced7cb1d3',
    family='export',
    contract=ShaderContract(
        class_name='ExportIndexCopyF32Program',
        shader_name='export_index_copy_f32_aced7cb1d3',
        fields=(
            TensorFieldSpec(
                name='cache',
                io_kind=IOKind.INOUT,
                role='state',
                contract=TensorContract(dtype='float32', shape=('B', 'H', 'S', 'D',)),
            ),
            TensorFieldSpec(
                name='index',
                io_kind=IOKind.INPUT,
                role='index',
                contract=TensorContract(dtype='int32', shape=('T',)),
            ),
            TensorFieldSpec(
                name='src',
                io_kind=IOKind.INPUT,
                role='input',
                contract=TensorContract(dtype='float32', shape=('B', 'H', 'T', 'D',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=20,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('H', PushConstantType.UINT32, 4, 'H', dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 8, 'S', dynamic=False),
                PushConstantFieldSpec('D', PushConstantType.UINT32, 12, 'D', dynamic=False),
                PushConstantFieldSpec('T', PushConstantType.UINT32, 16, 'T', dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul(mul('B', 'H'), 'T'), 'D'), 256), 1, 1),
    ),
    execution_requirements=None,
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict CacheBuffer { float cache[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndexBuffer { int index_values[]; };
layout(set = 0, binding = 2) buffer restrict readonly SrcBuffer { float src[]; };
layout(push_constant) uniform PushConstants { uint B; uint H; uint S; uint D; uint T; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.H * pc.T * pc.D;
    if (idx >= total) { return; }

    uint rem = idx;
    const uint d = rem % pc.D;
    rem = rem / pc.D;
    const uint t = rem % pc.T;
    rem = rem / pc.T;
    const uint h = rem % pc.H;
    const uint b = rem / pc.H;

    const uint dst_t = uint(index_values[t]);
    const uint src_idx = ((b * pc.H + h) * pc.T + t) * pc.D + d;
    const uint dst_idx = ((b * pc.H + h) * pc.S + dst_t) * pc.D + d;
    cache[dst_idx] = src[src_idx];
}
""",
)
