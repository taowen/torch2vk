"""Small runtime smoke shader that copies a Qwen3-ASR BF16 weight tensor."""

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
)
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements


WEIGHT_COPY_BF16 = ShaderVariant(
    name="qwen3_asr_weight_copy_bf16",
    family="qwen3_asr.smoke",
    contract=ShaderContract(
        class_name="Qwen3AsrWeightCopyBf16Program",
        shader_name="qwen3_asr_weight_copy_bf16",
        fields=(
            TensorFieldSpec(
                name="weight",
                io_kind=IOKind.INPUT,
                binding=0,
                role="weight",
                contract=TensorContract(dtype="bfloat16", shape=("N",)),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                binding=1,
                role="output",
                contract=TensorContract(dtype="bfloat16", shape=("N",)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(PushConstantFieldSpec("N", PushConstantType.UINT32, 0, "N"),),
        ),
        dispatch=(ceil_div("N", 64), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(set = 0, binding = 0) readonly buffer WeightBuffer {
    uint16_t weight[];
};

layout(set = 0, binding = 1) writeonly buffer OutputBuffer {
    uint16_t out_values[];
};

layout(push_constant) uniform Params {
    uint N;
} pc;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= pc.N) {
        return;
    }
    out_values[i] = weight[i];
}
""".lstrip(),
)
