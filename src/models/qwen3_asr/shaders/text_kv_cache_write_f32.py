"""Qwen3-ASR KV cache write shader for prefill and decode."""

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


QWEN3_ASR_TEXT_KV_CACHE_WRITE_F32 = ShaderVariant(
    name="qwen3_asr_text_kv_cache_write_f32",
    family="qwen3_asr.text",
    contract=ShaderContract(
        class_name="Qwen3AsrTextKvCacheWriteF32Program",
        shader_name="qwen3_asr_text_kv_cache_write_f32",
        fields=(
            TensorFieldSpec(
                name="k",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=(1, "T", "KH")),
            ),
            TensorFieldSpec(
                name="v",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=(1, "T", "KH")),
            ),
            TensorFieldSpec(
                name="key_cache",
                io_kind=IOKind.INOUT,
                role="state",
                contract=TensorContract(dtype="float32", shape=(1, "NK", "S", "D")),
            ),
            TensorFieldSpec(
                name="value_cache",
                io_kind=IOKind.INOUT,
                role="state",
                contract=TensorContract(dtype="float32", shape=(1, "NK", "S", "D")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=20,
            fields=(
                PushConstantFieldSpec("T", PushConstantType.UINT32, 0, "T"),
                PushConstantFieldSpec("num_kv_heads", PushConstantType.UINT32, 4, "NK"),
                PushConstantFieldSpec("head_dim", PushConstantType.UINT32, 8, "D"),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 12, "S"),
                PushConstantFieldSpec("cache_offset", PushConstantType.UINT32, 16, 0),
            ),
        ),
        dispatch=(ceil_div(mul("T", "KH"), 256), 1, 1),
    ),
    source="""
#version 450

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly KBuffer {
    float k_values[];
};

layout(set = 0, binding = 1) buffer restrict readonly VBuffer {
    float v_values[];
};

layout(set = 0, binding = 2) buffer restrict KeyCacheBuffer {
    float key_cache[];
};

layout(set = 0, binding = 3) buffer restrict ValueCacheBuffer {
    float value_cache[];
};

layout(push_constant) uniform PushConstants {
    uint T;
    uint num_kv_heads;
    uint head_dim;
    uint S;
    uint cache_offset;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint KH = pc.num_kv_heads * pc.head_dim;
    const uint total = pc.T * KH;
    if (index >= total) {
        return;
    }

    const uint token = index / KH;
    const uint within = index - token * KH;
    const uint head = within / pc.head_dim;
    const uint d = within - head * pc.head_dim;

    // cache layout: (1, NK, S, D) => cache[head * S * D + seq * D + d]
    const uint cache_idx = head * pc.S * pc.head_dim + (pc.cache_offset + token) * pc.head_dim + d;
    key_cache[cache_idx] = k_values[index];
    value_cache[cache_idx] = v_values[index];
}
""".lstrip(),
)
