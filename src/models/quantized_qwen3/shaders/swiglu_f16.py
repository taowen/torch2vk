"""Fused SiLU gate multiply for Qwen3 MLP."""

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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements


SWIGLU_F16 = ShaderVariant(
    name="swiglu_f16",
    family="quantized_qwen3",
    contract=ShaderContract(
        class_name="QuantizedQwen3SwiGLUF16Program",
        shader_name="swiglu_f16",
        fields=(
            TensorFieldSpec(
                name="gate",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float16", shape=("B", "T", "H")),
            ),
            TensorFieldSpec(
                name="up",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float16", shape=("B", "T", "H")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float16", shape=("B", "T", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec("N", PushConstantType.UINT32, 0, mul(mul("B", "T"), "H")),
            ),
        ),
        dispatch=(ceil_div(mul(mul("B", "T"), "H"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly GateBuffer { float16_t gate[]; };
layout(set = 0, binding = 1) buffer restrict readonly UpBuffer { float16_t up[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };

layout(push_constant) uniform PushConstants { uint N; } pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        const float x = float(gate[idx]);
        const float16_t silu = float16_t(x / (1.0 + exp(-x)));
        output_values[idx] = float16_t(silu * up[idx]);
    }
}
""",
)
