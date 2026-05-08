from __future__ import annotations

import hashlib

from torch.fx import Node

from torch2vk.export.shaders._factory import node_input_shape, node_output_shape
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

_SOURCE_DIM2_4D = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict CacheBuffer { float cache[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndexBuffer { int index_values[]; };
layout(set = 0, binding = 2) buffer restrict readonly SrcBuffer { float src[]; };
layout(push_constant) uniform PushConstants { uint B; uint H; uint S; uint D; uint T; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.H * pc.T * pc.D;
    if (idx >= total) { return; }

    uint rem = idx;
    const uint d = rem % pc.D;
    rem = rem / pc.D;
    const uint t = rem % pc.T;
    rem = rem / pc.T;
    const uint h = rem % pc.H;
    const uint b = rem / pc.H;

    const uint dst_t = uint(index_values[t]);
    const uint src_idx = ((b * pc.H + h) * pc.T + t) * pc.D + d;
    const uint dst_idx = ((b * pc.H + h) * pc.S + dst_t) * pc.D + d;
    cache[dst_idx] = src[src_idx];
}
"""

_SOURCE_DIM1_3D = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict CacheBuffer { float cache[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndexBuffer { int index_values[]; };
layout(set = 0, binding = 2) buffer restrict readonly SrcBuffer { float src[]; };
layout(push_constant) uniform PushConstants { uint B; uint T; uint H; uint S; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.S * pc.H;
    if (idx >= total) { return; }

    uint rem = idx;
    const uint h = rem % pc.H;
    rem = rem / pc.H;
    const uint s = rem % pc.S;
    const uint b = rem / pc.S;

    const uint dst_t = uint(index_values[s]);
    const uint src_idx = (b * pc.S + s) * pc.H + h;
    const uint dst_idx = (b * pc.T + dst_t) * pc.H + h;
    cache[dst_idx] = src[src_idx];
}
"""


_SOURCE_KV_CACHE_WRITE = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict CacheBuffer { float cache[]; };
layout(set = 0, binding = 1) buffer restrict readonly CachePositionBuffer { int cache_position[]; };
layout(set = 0, binding = 2) buffer restrict readonly SrcBuffer { float src[]; };
layout(push_constant) uniform PushConstants { uint B; uint H; uint S; uint D; uint T; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.H * pc.T * pc.D;
    if (idx >= total) { return; }

    uint rem = idx;
    const uint d = rem % pc.D;
    rem = rem / pc.D;
    const uint t = rem % pc.T;
    rem = rem / pc.T;
    const uint h = rem % pc.H;
    const uint b = rem / pc.H;

    const uint dst_t = uint(cache_position[t]);
    const uint src_idx = ((b * pc.H + h) * pc.T + t) * pc.D + d;
    const uint dst_idx = ((b * pc.H + h) * pc.S + dst_t) * pc.D + d;
    cache[dst_idx] = src[src_idx];
}
"""


def make_index_copy_variant(node: Node) -> ShaderVariant | None:
    if len(node.args) != 4:
        return None
    if _is_kv_cache_write(node):
        return _make_kv_cache_write_variant(node)
    dim = node.args[1]
    if dim == 1:
        return _make_index_copy_dim1_3d(node)
    if dim == 2:
        return _make_index_copy_dim2_4d(node)
    return None


def _is_kv_cache_write(node: Node) -> bool:
    return node.meta.get("torch2vk_kv_cache") in {
        "prefill_key_write",
        "prefill_value_write",
        "decode_key_write",
        "decode_value_write",
    }


def _make_kv_cache_write_variant(node: Node) -> ShaderVariant | None:
    dim = node.args[1]
    cache_shape = node_output_shape(node)
    index_shape = node_input_shape(node, 2)
    src_shape = node_input_shape(node, 3)
    if dim != 2:
        return None
    if len(cache_shape) != 4 or len(src_shape) != 4 or len(index_shape) != 1:
        return None
    if cache_shape[0] != src_shape[0] or cache_shape[1] != src_shape[1] or cache_shape[3] != src_shape[3]:
        return None
    if index_shape[0] != src_shape[2]:
        return None

    total = mul(mul(mul("B", "H"), "T"), "D")
    return ShaderVariant(
        name="export_kv_cache_write_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportKvCacheWriteF32Program",
            shader_name="export_kv_cache_write_f32",
            fields=(
                TensorFieldSpec("cache", IOKind.INOUT, "state", TensorContract(dtype="float32", shape=("B", "H", "S", "D"))),
                TensorFieldSpec("cache_position", IOKind.INPUT, "cache_position", TensorContract(dtype="int32", shape=("T",))),
                TensorFieldSpec("src", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("B", "H", "T", "D"))),
            ),
            push_constants=PushConstantSpec(
                size=20,
                fields=(
                    PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                    PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
                    PushConstantFieldSpec("S", PushConstantType.UINT32, 8, "S"),
                    PushConstantFieldSpec("D", PushConstantType.UINT32, 12, "D"),
                    PushConstantFieldSpec("T", PushConstantType.UINT32, 16, "T"),
                ),
            ),
            dispatch=(ceil_div(total, 256), 1, 1),
        ),
        source=_SOURCE_KV_CACHE_WRITE,
    )


def _make_index_copy_dim2_4d(node: Node) -> ShaderVariant | None:
    dim = node.args[1]
    cache_shape = node_output_shape(node)
    index_shape = node_input_shape(node, 2)
    src_shape = node_input_shape(node, 3)
    if len(cache_shape) != 4 or len(src_shape) != 4 or len(index_shape) != 1:
        return None
    if cache_shape[0] != src_shape[0] or cache_shape[1] != src_shape[1] or cache_shape[3] != src_shape[3]:
        return None
    if index_shape[0] != src_shape[2]:
        return None

    digest = hashlib.sha1(repr((cache_shape, index_shape, src_shape, dim)).encode()).hexdigest()[:10]
    shader_name = f"export_index_copy_f32_{digest}"
    total = mul(mul(mul("B", "H"), "T"), "D")
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name="ExportIndexCopyF32Program",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec("cache", IOKind.INOUT, "state", TensorContract(dtype="float32", shape=("B", "H", "S", "D"))),
                TensorFieldSpec("index", IOKind.INPUT, "index", TensorContract(dtype="int32", shape=("T",))),
                TensorFieldSpec("src", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("B", "H", "T", "D"))),
            ),
            push_constants=PushConstantSpec(
                size=20,
                fields=(
                    PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                    PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
                    PushConstantFieldSpec("S", PushConstantType.UINT32, 8, "S"),
                    PushConstantFieldSpec("D", PushConstantType.UINT32, 12, "D"),
                    PushConstantFieldSpec("T", PushConstantType.UINT32, 16, "T"),
                ),
            ),
            dispatch=(ceil_div(total, 256), 1, 1),
        ),
        source=_SOURCE_DIM2_4D,
    )


def _make_index_copy_dim1_3d(node: Node) -> ShaderVariant | None:
    dim = node.args[1]
    cache_shape = node_output_shape(node)
    index_shape = node_input_shape(node, 2)
    src_shape = node_input_shape(node, 3)
    if len(cache_shape) != 3 or len(src_shape) != 3 or len(index_shape) != 1:
        return None
    if cache_shape[0] != src_shape[0] or cache_shape[2] != src_shape[2]:
        return None
    if index_shape[0] != src_shape[1]:
        return None

    digest = hashlib.sha1(repr((cache_shape, index_shape, src_shape, dim)).encode()).hexdigest()[:10]
    shader_name = f"export_index_copy_f32_{digest}"
    total = mul(mul("B", "S"), "H")
    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name="ExportIndexCopyF32Program",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec("cache", IOKind.INOUT, "state", TensorContract(dtype="float32", shape=("B", "T", "H"))),
                TensorFieldSpec("index", IOKind.INPUT, "index", TensorContract(dtype="int32", shape=("S",))),
                TensorFieldSpec("src", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("B", "S", "H"))),
            ),
            push_constants=PushConstantSpec(
                size=16,
                fields=(
                    PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                    PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
                    PushConstantFieldSpec("H", PushConstantType.UINT32, 8, "H"),
                    PushConstantFieldSpec("S", PushConstantType.UINT32, 12, "S"),
                ),
            ),
            dispatch=(ceil_div(total, 256), 1, 1),
        ),
        source=_SOURCE_DIM1_3D,
    )
