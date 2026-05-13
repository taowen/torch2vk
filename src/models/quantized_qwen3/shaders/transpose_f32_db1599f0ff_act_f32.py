"""Transpose attention output from B,H,T,D to B,T,H,D using float32."""

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


TRANSPOSE_F32_DB1599F0FF_ACT_F32 = ShaderVariant(
    name="transpose_f32_db1599f0ff_act_f32",
    family="quantized_qwen3",
    contract=ShaderContract(
        class_name="TransposeF32Db1599F0FfActF32Program",
        shader_name="transpose_f32_db1599f0ff_act_f32",
        fields=(
            TensorFieldSpec("x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("B", "H", "T", "D"))),
            TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=("B", "T", "H", "D"))),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 8, "T"),
                PushConstantFieldSpec("D", PushConstantType.UINT32, 12, "D"),
            ),
        ),
        dispatch=(ceil_div(mul(mul(mul("B", "H"), "T"), "D"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint H; uint T; uint D; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.H * pc.T * pc.D;
    if (idx >= total) { return; }
    const uint d = idx % pc.D;
    const uint t = (idx / pc.D) % pc.T;
    const uint h = (idx / (pc.D * pc.T)) % pc.H;
    const uint b = idx / (pc.D * pc.T * pc.H);
    output_values[((b * pc.T + t) * pc.H + h) * pc.D + d] = x[idx];
}
""",
)
