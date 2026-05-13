"""SwiGLU for prefill float32 activations."""

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


SWIGLU_F32 = ShaderVariant(
    name="swiglu_f32",
    family="quantized_qwen3",
    contract=ShaderContract(
        class_name="SwigluF32Program",
        shader_name="swiglu_f32",
        fields=(
            TensorFieldSpec("gate", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("B", "T", "H"))),
            TensorFieldSpec("up", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("B", "T", "H"))),
            TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=("B", "T", "H"))),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, mul(mul("B", "T"), "H")),),
        ),
        dispatch=(ceil_div(mul(mul("B", "T"), "H"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly GateBuffer { float gate[]; };
layout(set = 0, binding = 1) buffer restrict readonly UpBuffer { float up[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const float g = gate[idx];
    output_values[idx] = (g / (1.0 + exp(-g))) * up[idx];
}
""",
)
