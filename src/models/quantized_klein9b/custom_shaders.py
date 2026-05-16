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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements


KLEIN9B_CAPTURE_QWEN3_CTX_F16 = ShaderVariant(
    name="klein9b_capture_qwen3_ctx_f16",
    family="klein9b",
    contract=ShaderContract(
        class_name="Klein9BCaptureQwen3CtxF16Program",
        shader_name="klein9b_capture_qwen3_ctx_f16",
        fields=(
            TensorFieldSpec(
                "layer_9",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "S", "H")),
            ),
            TensorFieldSpec(
                "layer_18",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "S", "H")),
            ),
            TensorFieldSpec(
                "layer_27",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "S", "H")),
            ),
            TensorFieldSpec(
                "ctx",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float16", shape=("B", "S", mul(3, "H"))),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("S", PushConstantType.UINT32, 0, "S"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
                PushConstantFieldSpec("N", PushConstantType.UINT32, 8, mul(mul("B", "S"), "H")),
            ),
        ),
        dispatch=(ceil_div(mul(mul("B", "S"), "H"), 256), 1, 1),
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
layout(set = 0, binding = 3) buffer restrict writeonly CtxBuffer { float16_t ctx[]; };
layout(push_constant) uniform PushConstants { uint S; uint H; uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const uint hidden = idx % pc.H;
    const uint token = (idx / pc.H) % pc.S;
    const uint batch = idx / (pc.H * pc.S);
    const uint out_base = (batch * pc.S + token) * (3u * pc.H) + hidden;
    ctx[out_base] = layer_9[idx];
    ctx[out_base + pc.H] = layer_18[idx];
    ctx[out_base + 2u * pc.H] = layer_27[idx];
}
""",
)


KLEIN9B_EULER_UPDATE_F16 = ShaderVariant(
    name="klein9b_euler_update_f16",
    family="klein9b",
    contract=ShaderContract(
        class_name="Klein9BEulerUpdateF16Program",
        shader_name="klein9b_euler_update_f16",
        fields=(
            TensorFieldSpec(
                "x",
                IOKind.INOUT,
                "state",
                TensorContract(dtype="float16", shape=("B", "S", "C")),
            ),
            TensorFieldSpec(
                "pred",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "S", "C")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("N", PushConstantType.UINT32, 0, mul(mul("B", "S"), "C")),
                PushConstantFieldSpec("dt", PushConstantType.FLOAT32, 4, PushConstantInput("dt")),
            ),
        ),
        dispatch=(ceil_div(mul(mul("B", "S"), "C"), 256), 1, 1),
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
