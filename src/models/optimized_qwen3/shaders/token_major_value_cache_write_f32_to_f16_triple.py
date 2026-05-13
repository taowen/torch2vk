"""Write token-major float32 values into three token-major float16 caches."""

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


TOKEN_MAJOR_VALUE_CACHE_WRITE_F32_TO_F16_TRIPLE = ShaderVariant(
    name="token_major_value_cache_write_f32_to_f16_triple",
    family="optimized_qwen3",
    contract=ShaderContract(
        class_name="OptimizedQwen3TokenMajorValueCacheWriteF32ToF16TripleProgram",
        shader_name="token_major_value_cache_write_f32_to_f16_triple",
        fields=(
            TensorFieldSpec(
                "cache_a",
                IOKind.INOUT,
                "state",
                TensorContract(dtype="float16", shape=("B", "SA", "H", "D")),
            ),
            TensorFieldSpec(
                "cache_b",
                IOKind.INOUT,
                "state",
                TensorContract(dtype="float16", shape=("B", "SB", "H", "D")),
            ),
            TensorFieldSpec(
                "cache_c",
                IOKind.INOUT,
                "state",
                TensorContract(dtype="float16", shape=("B", "SC", "H", "D")),
            ),
            TensorFieldSpec(
                "cache_position",
                IOKind.INPUT,
                "cache_position",
                TensorContract(dtype="int64", shape=("T",)),
            ),
            TensorFieldSpec(
                "src",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float32", shape=("B", "T", "H", "D")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=28,
            fields=(
                PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                PushConstantFieldSpec("SA", PushConstantType.UINT32, 4, "SA"),
                PushConstantFieldSpec("SB", PushConstantType.UINT32, 8, "SB"),
                PushConstantFieldSpec("SC", PushConstantType.UINT32, 12, "SC"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 16, "H"),
                PushConstantFieldSpec("D", PushConstantType.UINT32, 20, "D"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 24, "T"),
            ),
        ),
        dispatch=(ceil_div(mul(mul(mul("B", "T"), "H"), "D"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        require_shader_int64=True,
        require_storage_buffer_16bit_access=True,
    ),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict CacheABuffer { float16_t cache_a[]; };
layout(set = 0, binding = 1) buffer restrict CacheBBuffer { float16_t cache_b[]; };
layout(set = 0, binding = 2) buffer restrict CacheCBuffer { float16_t cache_c[]; };
layout(set = 0, binding = 3) buffer restrict readonly CachePositionBuffer { int64_t cache_position[]; };
layout(set = 0, binding = 4) buffer restrict readonly SrcBuffer { float src[]; };
layout(push_constant) uniform PushConstants { uint B; uint SA; uint SB; uint SC; uint H; uint D; uint T; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.T * pc.H * pc.D;
    if (idx >= total) { return; }

    uint rem = idx;
    const uint d = rem % pc.D;
    rem = rem / pc.D;
    const uint h = rem % pc.H;
    rem = rem / pc.H;
    const uint t = rem % pc.T;
    const uint b = rem / pc.T;

    const uint dst_t = uint(cache_position[t]);
    const uint src_idx = ((b * pc.T + t) * pc.H + h) * pc.D + d;
    const float16_t value = float16_t(src[src_idx]);
    if (dst_t < pc.SA) {
        cache_a[((b * pc.SA + dst_t) * pc.H + h) * pc.D + d] = value;
    }
    if (dst_t < pc.SB) {
        cache_b[((b * pc.SB + dst_t) * pc.H + h) * pc.D + d] = value;
    }
    if (dst_t < pc.SC) {
        cache_c[((b * pc.SC + dst_t) * pc.H + h) * pc.D + d] = value;
    }
}
""",
)
