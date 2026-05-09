from __future__ import annotations

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
)
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements, SubgroupRequirements

_SOURCE_CAUSAL = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float v[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint NH; uint NK; uint T; uint S; uint D; } pc;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint batch_head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint d_out = gl_WorkGroupID.z * 64u + gl_LocalInvocationID.x;
    if (batch_head >= pc.B * pc.NH || row >= pc.T || d_out >= pc.D) { return; }
    const uint batch = batch_head / pc.NH;
    const uint head = batch_head % pc.NH;
    const uint kv_head = head * pc.NK / pc.NH;
    const uint q_base = (batch * pc.NH + head) * pc.T * pc.D;
    const uint k_base = (batch * pc.NK + kv_head) * pc.S * pc.D;
    const uint v_base = k_base;
    const float scale = inversesqrt(float(pc.D));
    float max_score = -1.0e38;
    for (uint col = 0u; col <= row && col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[q_base + row * pc.D + d] * k[k_base + col * pc.D + d];
        }
        max_score = max(max_score, dot * scale);
    }
    float sum_exp = 0.0;
    for (uint col = 0u; col <= row && col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[q_base + row * pc.D + d] * k[k_base + col * pc.D + d];
        }
        sum_exp += exp(dot * scale - max_score);
    }
    float acc = 0.0;
    for (uint col = 0u; col <= row && col < pc.S; ++col) {
        float dot = 0.0;
        for (uint dd = 0u; dd < pc.D; ++dd) {
            dot += q[q_base + row * pc.D + dd] * k[k_base + col * pc.D + dd];
        }
        float w = exp(dot * scale - max_score) / sum_exp;
        acc += w * v[v_base + col * pc.D + d_out];
    }
    output_values[(batch * pc.NH + head) * pc.T * pc.D + row * pc.D + d_out] = acc;
}
"""

_SOURCE_NONCAUSAL = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float v[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint NH; uint NK; uint T; uint S; uint D; } pc;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint batch_head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint d_out = gl_WorkGroupID.z * 64u + gl_LocalInvocationID.x;
    if (batch_head >= pc.B * pc.NH || row >= pc.T || d_out >= pc.D) { return; }
    const uint batch = batch_head / pc.NH;
    const uint head = batch_head % pc.NH;
    const uint kv_head = head * pc.NK / pc.NH;
    const uint q_base = (batch * pc.NH + head) * pc.T * pc.D;
    const uint k_base = (batch * pc.NK + kv_head) * pc.S * pc.D;
    const uint v_base = k_base;
    const float scale = inversesqrt(float(pc.D));
    float max_score = -1.0e38;
    for (uint col = 0u; col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[q_base + row * pc.D + d] * k[k_base + col * pc.D + d];
        }
        max_score = max(max_score, dot * scale);
    }
    float sum_exp = 0.0;
    for (uint col = 0u; col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[q_base + row * pc.D + d] * k[k_base + col * pc.D + d];
        }
        sum_exp += exp(dot * scale - max_score);
    }
    float acc = 0.0;
    for (uint col = 0u; col < pc.S; ++col) {
        float dot = 0.0;
        for (uint dd = 0u; dd < pc.D; ++dd) {
            dot += q[q_base + row * pc.D + dd] * k[k_base + col * pc.D + dd];
        }
        float w = exp(dot * scale - max_score) / sum_exp;
        acc += w * v[v_base + col * pc.D + d_out];
    }
    output_values[(batch * pc.NH + head) * pc.T * pc.D + row * pc.D + d_out] = acc;
}
"""

_SOURCE_MASKED = """\
#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float v[]; };
layout(set = 0, binding = 3) buffer restrict readonly MaskBuffer { float mask[]; };
layout(set = 0, binding = 4) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint B; uint NH; uint NK; uint T; uint S; uint D; } pc;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const float NEG_INF = -3.4028234663852886e38;

void main() {
    const uint batch_head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint dim0 = gl_LocalInvocationID.x;
    const uint dim1 = dim0 + 64u;
    if (batch_head >= pc.B * pc.NH || row >= pc.T) { return; }
    const bool valid0 = dim0 < pc.D;
    const bool valid1 = dim1 < pc.D;

    const uint batch = batch_head / pc.NH;
    const uint head = batch_head % pc.NH;
    const uint kv_head = head * pc.NK / pc.NH;
    const uint q_base = (batch * pc.NH + head) * pc.T * pc.D;
    const uint k_base = (batch * pc.NK + kv_head) * pc.S * pc.D;
    const uint v_base = k_base;
    const uint mask_base = batch * pc.T * pc.S;
    const uint q_row_base = q_base + row * pc.D;
    const float q0 = valid0 ? q[q_row_base + dim0] : 0.0;
    const float q1 = valid1 ? q[q_row_base + dim1] : 0.0;
    const float scale = inversesqrt(float(pc.D));

    float running_max = NEG_INF;
    float running_sum = 0.0;
    float acc0 = 0.0;
    float acc1 = 0.0;

    for (uint col = 0u; col < pc.S; ++col) {
        const uint kv_offset = col * pc.D;
        const float k0 = valid0 ? k[k_base + kv_offset + dim0] : 0.0;
        const float k1 = valid1 ? k[k_base + kv_offset + dim1] : 0.0;
        const float dot = subgroupAdd(q0 * k0 + q1 * k1);
        const float score = dot * scale + mask[mask_base + row * pc.S + col];
        const float next_max = max(running_max, score);
        const float old_scale = running_max == NEG_INF ? 0.0 : exp(running_max - next_max);
        const float score_scale = exp(score - next_max);
        if (valid0) {
            acc0 = acc0 * old_scale + score_scale * v[v_base + kv_offset + dim0];
        }
        if (valid1) {
            acc1 = acc1 * old_scale + score_scale * v[v_base + kv_offset + dim1];
        }
        running_sum = running_sum * old_scale + score_scale;
        running_max = next_max;
    }

    if (running_sum > 0.0) {
        const uint output_base = (batch * pc.NH + head) * pc.T * pc.D + row * pc.D;
        if (valid0) {
            output_values[output_base + dim0] = acc0 / running_sum;
        }
        if (valid1) {
            output_values[output_base + dim1] = acc1 / running_sum;
        }
    }
}
"""


_SOURCE_DECODE_CACHE = """\
#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
{{CACHE_POSITION_EXTENSION}}

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float v[]; };
layout(set = 0, binding = 3) buffer restrict readonly CachePositionBuffer { {{CACHE_POSITION_TYPE}} cache_position[]; };
layout(set = 0, binding = 4) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint NH; uint NK; uint S; uint D; } pc;
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

const float NEG_INF = -3.4028234663852886e38;

shared float subgroup_dot[4];

void main() {
    const uint head = gl_WorkGroupID.x;
    const uint dim = gl_LocalInvocationID.x;
    if (head >= pc.NH) { return; }
    const bool valid_dim = dim < pc.D;

    const uint kv_head = head * pc.NK / pc.NH;
    const float q_value = valid_dim ? q[head * pc.D + dim] : 0.0;
    const float scale = inversesqrt(float(pc.D));

    float running_max = NEG_INF;
    float running_sum = 0.0;
    float acc = 0.0;

    const uint cache_head_base = kv_head * pc.S * pc.D;
    const uint cache_len = min(uint(cache_position[0]) + 1u, pc.S);
    for (uint key_pos = 0u; key_pos < cache_len; ++key_pos) {
        const float k_val = valid_dim ? k[cache_head_base + key_pos * pc.D + dim] : 0.0;
        const float v_val = valid_dim ? v[cache_head_base + key_pos * pc.D + dim] : 0.0;
        barrier();

        const float dot_part = valid_dim ? q_value * k_val : 0.0;
        const float dot_sum = subgroupAdd(dot_part);
        if (gl_SubgroupInvocationID == 0u) {
            subgroup_dot[dim / gl_SubgroupSize] = dot_sum;
        }
        barrier();

        if (valid_dim) {
            float dot = 0.0;
            for (uint i = 0u; i < (pc.D + gl_SubgroupSize - 1u) / gl_SubgroupSize; ++i) {
                dot += subgroup_dot[i];
            }
            const float score = dot * scale;
            const float next_max = max(running_max, score);
            const float old_scale = running_max == NEG_INF ? 0.0 : exp(running_max - next_max);
            const float score_scale = exp(score - next_max);
            acc = acc * old_scale + score_scale * v_val;
            running_sum = running_sum * old_scale + score_scale;
            running_max = next_max;
        }
        barrier();
    }

    if (valid_dim && running_sum > 0.0) {
        output_values[head * pc.D + dim] = acc / running_sum;
    }
}
"""


def _is_causal(node: Node) -> bool:
    if len(node.args) >= 6 and isinstance(node.args[5], bool):
        return node.args[5]
    return False


def _has_mask(node: Node) -> bool:
    if len(node.args) < 4:
        return False
    mask_arg = node.args[3]
    if not isinstance(mask_arg, Node):
        return False
    tm = mask_arg.meta.get("tensor_meta")
    return tm is not None


def make_sdpa_variant(node: Node) -> ShaderVariant | None:
    q_shape = node_input_shape(node, 0)
    k_shape = node_input_shape(node, 1)
    v_shape = node_input_shape(node, 2)
    out_shape = node_output_shape(node)
    if not q_shape or not k_shape or not v_shape or not out_shape:
        return None

    q_contract = tuple(f"Q{i}" for i in range(len(q_shape)))
    k_contract = tuple(f"K{i}" for i in range(len(k_shape)))
    v_contract = tuple(f"V{i}" for i in range(len(v_shape)))
    out_contract = tuple(f"O{i}" for i in range(len(out_shape)))

    b = q_shape[0] if len(q_shape) >= 4 else 1
    nh = q_shape[len(q_shape) - 3] if len(q_shape) >= 3 else 1
    nk = k_shape[len(k_shape) - 3] if len(k_shape) >= 3 else 1
    t = q_shape[len(q_shape) - 2] if len(q_shape) >= 2 else 1
    s = k_shape[len(k_shape) - 2] if len(k_shape) >= 2 else 1
    d = q_shape[len(q_shape) - 1]

    if node.meta.get("torch2vk_kv_cache") == "decode_attention":
        cache_position_dtype = str(node.meta.get("torch2vk_cache_position_dtype", ""))
        return _make_decode_cache_variant(
            q_shape,
            k_shape,
            v_shape,
            out_shape,
            nh,
            nk,
            t,
            s,
            d,
            cache_position_dtype,
        )

    causal = _is_causal(node)
    masked = _has_mask(node)

    execution_requirements = None
    if causal:
        source = _SOURCE_CAUSAL
        shader_name = "sdpa_causal_f32"
    elif masked:
        if d > 128:
            return None
        source = _SOURCE_MASKED
        shader_name = "sdpa_masked_f32"
        execution_requirements = ShaderExecutionRequirements(
            subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
        )
    else:
        source = _SOURCE_NONCAUSAL
        shader_name = "sdpa_f32"

    if masked:
        mask_shape = node_input_shape(node, 3)
        mask_contract = tuple(f"M{i}" for i in range(len(mask_shape)))
        fields = (
            TensorFieldSpec("q", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=q_contract)),
            TensorFieldSpec("k", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=k_contract)),
            TensorFieldSpec("v", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=v_contract)),
            TensorFieldSpec("mask", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=mask_contract)),
            TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
        )
    else:
        fields = (
            TensorFieldSpec("q", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=q_contract)),
            TensorFieldSpec("k", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=k_contract)),
            TensorFieldSpec("v", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=v_contract)),
            TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
        )

    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name="ExportSdpaProgram",
            shader_name=shader_name,
            fields=fields,
            push_constants=PushConstantSpec(
                size=24,
                fields=(
                    PushConstantFieldSpec("B", PushConstantType.UINT32, 0, b),
                    PushConstantFieldSpec("NH", PushConstantType.UINT32, 4, nh),
                    PushConstantFieldSpec("NK", PushConstantType.UINT32, 8, nk),
                    PushConstantFieldSpec("T", PushConstantType.UINT32, 12, t),
                    PushConstantFieldSpec("S", PushConstantType.UINT32, 16, s),
                    PushConstantFieldSpec("D", PushConstantType.UINT32, 20, d),
                ),
            ),
            dispatch=(b * nh, t, 1 if masked else (d + 63) // 64),
        ),
        execution_requirements=execution_requirements,
        source=source,
    )


def _make_decode_cache_variant(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    nh: int,
    nk: int,
    t: int,
    s: int,
    d: int,
    cache_position_dtype: str,
) -> ShaderVariant | None:
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4 or len(out_shape) != 4:
        return None
    if t != 1 or d > 128:
        return None
    if cache_position_dtype not in {"int32", "int64"}:
        return None
    return ShaderVariant(
        name="sdpa_decode_cache_f32",
        family="export",
        contract=ShaderContract(
            class_name="ExportSdpaDecodeCacheF32Program",
            shader_name="sdpa_decode_cache_f32",
            fields=(
                TensorFieldSpec("q", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=("B", "NH", "T", "D"))),
                TensorFieldSpec("k", IOKind.INPUT, "state", TensorContract(dtype="float32", shape=("B", "NK", "S", "D"))),
                TensorFieldSpec("v", IOKind.INPUT, "state", TensorContract(dtype="float32", shape=("B", "NK", "S", "D"))),
                TensorFieldSpec("cache_position", IOKind.INPUT, "cache_position", TensorContract(dtype=cache_position_dtype, shape=("T",))),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=("B", "NH", "T", "D"))),
            ),
            push_constants=PushConstantSpec(
                size=16,
                fields=(
                    PushConstantFieldSpec("NH", PushConstantType.UINT32, 0, nh),
                    PushConstantFieldSpec("NK", PushConstantType.UINT32, 4, nk),
                    PushConstantFieldSpec("S", PushConstantType.UINT32, 8, s),
                    PushConstantFieldSpec("D", PushConstantType.UINT32, 12, d),
                ),
            ),
            dispatch=(nh, 1, 1),
        ),
        execution_requirements=ShaderExecutionRequirements(
            subgroup=SubgroupRequirements(required_size=64, require_full_subgroups=True),
            require_shader_int64=cache_position_dtype == "int64",
        ),
        source=_decode_cache_source(cache_position_dtype),
    )


def _decode_cache_source(cache_position_dtype: str) -> str:
    cache_position_type = "int64_t" if cache_position_dtype == "int64" else "int"
    extension = (
        "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require"
        if cache_position_dtype == "int64"
        else ""
    )
    return (
        _SOURCE_DECODE_CACHE
        .replace("{{CACHE_POSITION_EXTENSION}}", extension)
        .replace("{{CACHE_POSITION_TYPE}}", cache_position_type)
    )
