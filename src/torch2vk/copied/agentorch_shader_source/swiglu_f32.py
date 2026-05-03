"""Exact llama.cpp swiglu_f32 shader binding."""

from __future__ import annotations

from agentorch.kernel.contract import ceil_div, input_tensor, mul, output_tensor, shader_contract, storage_buffer_binding

from .llama_push_constants import glu_push_constant_block
from .shader_variant import shader_variant


_SWIGLU_COMP_SOURCE = """
#version 450


#extension GL_EXT_shader_16bit_storage : require


layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer A { float data_a[]; };
layout(binding = 1) readonly buffer B { float data_b[]; };
layout(binding = 2) writeonly buffer D { float data_d[]; };

layout(push_constant) uniform parameter
{
    uint N;
    uint ne00;
    uint ne20;
    uint mode;
    float alpha;
    float limit;
    uint nb01;
    uint nb02;
    uint nb03;
    uint ne01;
    uint ne02;
    uint nb11;
    uint nb12;
    uint nb13;
    uint ne11;
    uint ne12;
} p;

float op(float a, float b) {
    return a / (1.0f + exp(- a)) * b;
}

 void main() {
    const uint i = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;

    if (i >= p.N) {
        return;
    }

    const uint row = i / p.ne20;
    const uint col = i - row * p.ne20;

    const uint i3 = row / (p.ne01 * p.ne02);
    const uint i2 = (row % (p.ne01 * p.ne02)) / p.ne01;
    const uint i1 = row % p.ne01;
    const uint src_idx = i3 * p.nb03 + i2 * p.nb02 + i1 * p.nb01 + col;

    const uint dst_i3 = row / (p.ne11 * p.ne12);
    const uint dst_i2 = (row % (p.ne11 * p.ne12)) / p.ne11;
    const uint dst_i1 = row % p.ne11;
    const uint dst_idx = dst_i3 * p.nb13 + dst_i2 * p.nb12 + dst_i1 * p.nb11 + col;

    if (p.mode == 0) {

        const uint offset = p.ne00 / 2;
        const uint idx = src_idx;

        data_d[dst_idx] = float(op(float(data_a[idx]), float(data_a[idx + offset])));
    } else if (p.mode == 1) {

        const uint offset = p.ne00 / 2;
        const uint idx = src_idx;

        data_d[dst_idx] = float(op(float(data_a[idx + offset]), float(data_a[idx])));
    } else {

        const uint idx = src_idx;

        data_d[dst_idx] = float(op(float(data_a[idx]), float(data_b[idx])));
    }
}
"""


SWIGLU_F32 = shader_variant(
    name="swiglu_f32",
    family="swiglu_f32",
    contract=shader_contract(
        class_name="SwiGluF32Program",
        shader_name="swiglu_f32",
        fields=(
            input_tensor(
                name="gate",
                binding="t_gate",
                role="gate",
                dtypes=("float32",),
                shape=("B", "S", "H"),
            ),
            input_tensor(
                name="up",
                binding="t_up",
                role="up",
                dtypes=("float32",),
                shape=("B", "S", "H"),
            ),
            output_tensor(
                name="output",
                binding="t_output",
                role="output",
                dtypes=("float32",),
                shape=("B", "S", "H"),
            ),
        ),
        uniforms=(),
        push_constants=glu_push_constant_block(src0_name="gate", dst_name="output", mode=2),
        dispatch=(1, ceil_div(mul(mul("B", "S"), "H"), 512), 1),
        bindings=(
            storage_buffer_binding(name="t_gate", binding=0),
            storage_buffer_binding(name="t_up", binding=1),
            storage_buffer_binding(name="t_output", binding=2),
        ),
    ),
    source=_SWIGLU_COMP_SOURCE,
)
