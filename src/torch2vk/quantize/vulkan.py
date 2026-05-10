"""Vulkan compute quantizers for GGUF tensor blocks."""

from __future__ import annotations

import numpy as np

from torch2vk.runtime.logical import LogicalTensor, MemoryClass, TensorLifetime, TensorRole
from torch2vk.runtime.session import RuntimeSession
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
from torch2vk.vulkan.types import TensorSpec


def quantize_q8_0_vulkan(rt: RuntimeSession, array: np.ndarray, *, name: str) -> np.ndarray:
    source = _source_tensor(name, array.shape)
    blocks_per_row = array.shape[1] // 32
    packed = _output_tensor(f"{name}.q8_0", "uint16", (array.shape[0], blocks_per_row * 17))
    rt.register_inputs({source: np.ascontiguousarray(array, dtype=np.float32)})
    with rt.frame(f"gguf_quantize.q8_0.{name}"):
        GGUF_QUANTIZE_Q8_0_F32(rt, x=source, output=packed)
        words = rt.readback(packed)
    return np.ascontiguousarray(words.view(np.uint8).reshape(array.shape[0], blocks_per_row * 34))


def quantize_q4_k_vulkan(rt: RuntimeSession, array: np.ndarray, *, name: str) -> np.ndarray:
    source = _source_tensor(name, array.shape)
    blocks_per_row = array.shape[1] // 256
    packed = _output_tensor(f"{name}.q4_k", "uint32", (array.shape[0], blocks_per_row * 36))
    rt.register_inputs({source: np.ascontiguousarray(array, dtype=np.float32)})
    with rt.frame(f"gguf_quantize.q4_k.{name}"):
        GGUF_QUANTIZE_Q4_K_F32(rt, x=source, output=packed)
        words = rt.readback(packed)
    return np.ascontiguousarray(words.view(np.uint8).reshape(array.shape[0], blocks_per_row * 144))


def _source_tensor(name: str, shape: tuple[int, ...]) -> LogicalTensor:
    if len(shape) != 2:
        raise ValueError(f"{name} quantization expects rank-2 tensor, got {shape}")
    return LogicalTensor(
        spec=TensorSpec(dtype="float32", shape=shape),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
        name=f"gguf_quantize.{name}.source",
    )


def _output_tensor(name: str, dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.OUTPUT,
        memory=MemoryClass.HOST_OUTPUT,
        lifetime=TensorLifetime.FRAME,
        name=f"gguf_quantize.{name}.output",
    )


GGUF_QUANTIZE_Q8_0_F32 = ShaderVariant(
    name="gguf_quantize_q8_0_f32",
    family="gguf_quantize",
    contract=ShaderContract(
        class_name="GGUFQuantizeQ8_0F32Program",
        shader_name="gguf_quantize_q8_0_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("R", "K")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="uint16", shape=("R", mul(ceil_div("K", 32), 17))),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("R", PushConstantType.UINT32, 0, "R"),
                PushConstantFieldSpec("K", PushConstantType.UINT32, 4, "K"),
            ),
        ),
        dispatch=("R", ceil_div("K", 32), 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { uint16_t output_values[]; };

layout(push_constant) uniform PushConstants {
    uint R;
    uint K;
} pc;

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

shared float block_values[32];
shared int block_quants[32];
shared float block_scale;

float round_away(float value) {
    const float absolute = abs(value);
    const float floored = floor(absolute);
    const float rounded = floored + floor(2.0 * (absolute - floored));
    return sign(value) * rounded;
}

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint block = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const uint k = block * 32u + lane;
    const float value = x[row * pc.K + k];
    block_values[lane] = value;
    barrier();

    if (lane == 0u) {
        float amax = 0.0;
        for (uint i = 0u; i < 32u; ++i) {
            amax = max(amax, abs(block_values[i]));
        }
        block_scale = amax / 127.0;
    }
    barrier();

    const float inv_scale = block_scale == 0.0 ? 0.0 : 1.0 / block_scale;
    int quant = int(round_away(value * inv_scale));
    quant = clamp(quant, -128, 127);
    block_quants[lane] = quant;
    barrier();

    const uint output_base = row * (pc.K / 32u) * 17u + block * 17u;
    if (lane == 0u) {
        output_values[output_base] = uint16_t(packHalf2x16(vec2(block_scale, 0.0)) & 0xffffu);
    }
    if ((lane & 1u) == 0u) {
        const uint lo = uint(block_quants[lane] & 255);
        const uint hi = uint(block_quants[lane + 1u] & 255);
        output_values[output_base + 1u + (lane >> 1u)] = uint16_t(lo | (hi << 8u));
    }
}
""",
)


GGUF_QUANTIZE_Q4_K_F32 = ShaderVariant(
    name="gguf_quantize_q4_k_f32",
    family="gguf_quantize",
    contract=ShaderContract(
        class_name="GGUFQuantizeQ4KF32Program",
        shader_name="gguf_quantize_q4_k_f32",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=("R", "K")),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="uint32", shape=("R", mul(ceil_div("K", 256), 36))),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("R", PushConstantType.UINT32, 0, "R"),
                PushConstantFieldSpec("K", PushConstantType.UINT32, 4, "K"),
            ),
        ),
        dispatch=("R", ceil_div("K", 256), 1),
    ),
    source="""\
#version 450

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { uint output_values[]; };

layout(push_constant) uniform PushConstants {
    uint R;
    uint K;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared float values[256];
shared float subblock_scales[8];
shared float subblock_mins[8];
shared uint scale_codes[8];
shared uint min_codes[8];
shared uint quants[256];
shared float d_scale;
shared float d_min;

uint rounded_code(float value, uint max_code) {
    return uint(clamp(floor(value + 0.5), 0.0, float(max_code)));
}

float weighted_quant_error(uint base, float scale, float minimum, float inv_scale) {
    float error = 0.0;
    float sum_x2 = 0.0;
    for (uint i = 0u; i < 32u; ++i) {
        const float value = values[base + i];
        sum_x2 += value * value;
    }
    const float av_x = sqrt(sum_x2 / 32.0);
    for (uint i = 0u; i < 32u; ++i) {
        const float value = values[base + i];
        const float weight = av_x + abs(value);
        const float level = float(rounded_code(inv_scale * (value - minimum), 15u));
        const float diff = scale * level + minimum - value;
        error += weight * diff * diff;
    }
    return error;
}

float choose_subblock_scale(uint base, out float min_offset) {
    float min_value = values[base];
    float max_value = values[base];
    float sum_x2 = 0.0;
    for (uint i = 0u; i < 32u; ++i) {
        const float value = values[base + i];
        min_value = min(min_value, value);
        max_value = max(max_value, value);
        sum_x2 += value * value;
    }
    if (min_value > 0.0) {
        min_value = 0.0;
    }
    if (max_value == min_value) {
        min_offset = -min_value;
        return 0.0;
    }

    const float av_x = sqrt(sum_x2 / 32.0);
    float sum_w = 0.0;
    float sum_x = 0.0;
    for (uint i = 0u; i < 32u; ++i) {
        const float value = values[base + i];
        const float weight = av_x + abs(value);
        sum_w += weight;
        sum_x += weight * value;
    }

    float inv_scale = 15.0 / (max_value - min_value);
    float best_scale = 1.0 / inv_scale;
    float best_min = min_value;
    float best_error = weighted_quant_error(base, best_scale, best_min, inv_scale);

    for (uint step = 0u; step <= 20u; ++step) {
        inv_scale = (14.0 + 0.1 * float(step)) / (max_value - min_value);
        float sum_l = 0.0;
        float sum_l2 = 0.0;
        float sum_xl = 0.0;
        for (uint i = 0u; i < 32u; ++i) {
            const float value = values[base + i];
            const float weight = av_x + abs(value);
            const float level = float(rounded_code(inv_scale * (value - min_value), 15u));
            sum_l += weight * level;
            sum_l2 += weight * level * level;
            sum_xl += weight * level * value;
        }

        const float determinant = sum_w * sum_l2 - sum_l * sum_l;
        if (determinant > 0.0) {
            float candidate_scale = (sum_w * sum_xl - sum_x * sum_l) / determinant;
            float candidate_min = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
            if (candidate_min > 0.0) {
                candidate_min = 0.0;
                candidate_scale = sum_l2 > 0.0 ? sum_xl / sum_l2 : 0.0;
            }
            const float candidate_error = weighted_quant_error(base, candidate_scale, candidate_min, inv_scale);
            if (candidate_error < best_error) {
                best_error = candidate_error;
                best_scale = candidate_scale;
                best_min = candidate_min;
            }
        }
    }

    min_offset = max(0.0, -best_min);
    return max(0.0, best_scale);
}

uint metadata_byte(uint byte_offset) {
    if (byte_offset < 4u) {
        const uint i = byte_offset;
        return (scale_codes[i] & 63u) | ((scale_codes[i + 4u] & 48u) << 2u);
    }
    if (byte_offset < 8u) {
        const uint i = byte_offset - 4u;
        return (min_codes[i] & 63u) | ((min_codes[i + 4u] & 48u) << 2u);
    }
    const uint i = byte_offset - 8u;
    return (scale_codes[i + 4u] & 15u) | ((min_codes[i + 4u] & 15u) << 4u);
}

uint q_byte(uint byte_offset) {
    const uint group = byte_offset >> 5u;
    const uint local = byte_offset & 31u;
    const uint low = quants[group * 64u + local] & 15u;
    const uint high = quants[group * 64u + 32u + local] & 15u;
    return low | (high << 4u);
}

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint block = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const uint row_offset = row * pc.K + block * 256u;
    values[lane] = x[row_offset + lane];
    barrier();

    if (lane < 8u) {
        const uint base = lane * 32u;
        float min_offset = 0.0;
        subblock_scales[lane] = choose_subblock_scale(base, min_offset);
        subblock_mins[lane] = min_offset;
    }
    barrier();

    if (lane == 0u) {
        float max_scale = 0.0;
        float max_min = 0.0;
        for (uint i = 0u; i < 8u; ++i) {
            max_scale = max(max_scale, subblock_scales[i]);
            max_min = max(max_min, subblock_mins[i]);
        }
        d_scale = max_scale > 0.0 ? max_scale / 63.0 : 0.0;
        d_min = max_min > 0.0 ? max_min / 63.0 : 0.0;
        const float inv_scale = max_scale > 0.0 ? 63.0 / max_scale : 0.0;
        const float inv_min = max_min > 0.0 ? 63.0 / max_min : 0.0;
        for (uint i = 0u; i < 8u; ++i) {
            scale_codes[i] = rounded_code(subblock_scales[i] * inv_scale, 63u);
            min_codes[i] = rounded_code(subblock_mins[i] * inv_min, 63u);
        }
        d_scale = unpackHalf2x16(packHalf2x16(vec2(d_scale, 0.0))).x;
        d_min = unpackHalf2x16(packHalf2x16(vec2(d_min, 0.0))).x;
    }
    barrier();

    const uint subblock = lane >> 5u;
    const float scale = d_scale * float(scale_codes[subblock]);
    const float minimum = d_min * float(min_codes[subblock]);
    uint quant = 0u;
    if (scale > 0.0) {
        quant = rounded_code((values[lane] + minimum) / scale, 15u);
    }
    quants[lane] = quant;
    barrier();

    const uint output_base = row * (pc.K / 256u) * 36u + block * 36u;
    if (lane == 0u) {
        output_values[output_base] = packHalf2x16(vec2(d_scale, d_min));
    } else if (lane < 4u) {
        const uint word = lane;
        const uint byte0 = metadata_byte((word - 1u) * 4u + 0u);
        const uint byte1 = metadata_byte((word - 1u) * 4u + 1u);
        const uint byte2 = metadata_byte((word - 1u) * 4u + 2u);
        const uint byte3 = metadata_byte((word - 1u) * 4u + 3u);
        output_values[output_base + word] = byte0 | (byte1 << 8u) | (byte2 << 16u) | (byte3 << 24u);
    } else if (lane < 36u) {
        const uint word = lane;
        const uint q_offset = (word - 4u) * 4u;
        const uint byte0 = q_byte(q_offset + 0u);
        const uint byte1 = q_byte(q_offset + 1u);
        const uint byte2 = q_byte(q_offset + 2u);
        const uint byte3 = q_byte(q_offset + 3u);
        output_values[output_base + word] = byte0 | (byte1 << 8u) | (byte2 << 16u) | (byte3 << 24u);
    }
}
""",
)
