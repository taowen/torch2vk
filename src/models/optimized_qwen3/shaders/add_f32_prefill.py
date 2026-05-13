"""Elementwise add for prefill float32 activations."""

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


ADD_F32_PREFILL = ShaderVariant(
    name="add_f32_prefill",
    family="optimized_qwen3",
    contract=ShaderContract(
        class_name="AddF32PrefillProgram",
        shader_name="add_f32_prefill",
        fields=(
            TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=(1, "T", "H"))),
            TensorFieldSpec("y", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=(1, "T", "H"))),
            TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=(1, "T", "H"))),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, mul("T", "H")),),
        ),
        dispatch=(ceil_div(mul("T", "H"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) { output_values[idx] = x[idx] + y[idx]; }
}
""",
)
