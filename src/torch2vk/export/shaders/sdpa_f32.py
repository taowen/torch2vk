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

_SOURCE_CAUSAL = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float v[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint NH; uint NK; uint T; uint S; uint D; } pc;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint d_out = gl_WorkGroupID.z * 64u + gl_LocalInvocationID.x;
    if (head >= pc.NH || row >= pc.T || d_out >= pc.D) { return; }
    const uint kv_head = head * pc.NK / pc.NH;
    const float scale = inversesqrt(float(pc.D));
    float max_score = -1.0e38;
    for (uint col = 0u; col <= row && col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[(head * pc.T + row) * pc.D + d] * k[(kv_head * pc.S + col) * pc.D + d];
        }
        max_score = max(max_score, dot * scale);
    }
    float sum_exp = 0.0;
    for (uint col = 0u; col <= row && col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[(head * pc.T + row) * pc.D + d] * k[(kv_head * pc.S + col) * pc.D + d];
        }
        sum_exp += exp(dot * scale - max_score);
    }
    float acc = 0.0;
    for (uint col = 0u; col <= row && col < pc.S; ++col) {
        float dot = 0.0;
        for (uint dd = 0u; dd < pc.D; ++dd) {
            dot += q[(head * pc.T + row) * pc.D + dd] * k[(kv_head * pc.S + col) * pc.D + dd];
        }
        float w = exp(dot * scale - max_score) / sum_exp;
        acc += w * v[(kv_head * pc.S + col) * pc.D + d_out];
    }
    output_values[(head * pc.T + row) * pc.D + d_out] = acc;
}
"""

_SOURCE_NONCAUSAL = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float v[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint NH; uint NK; uint T; uint S; uint D; } pc;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint d_out = gl_WorkGroupID.z * 64u + gl_LocalInvocationID.x;
    if (head >= pc.NH || row >= pc.T || d_out >= pc.D) { return; }
    const uint kv_head = head * pc.NK / pc.NH;
    const float scale = inversesqrt(float(pc.D));
    float max_score = -1.0e38;
    for (uint col = 0u; col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[(head * pc.T + row) * pc.D + d] * k[(kv_head * pc.S + col) * pc.D + d];
        }
        max_score = max(max_score, dot * scale);
    }
    float sum_exp = 0.0;
    for (uint col = 0u; col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[(head * pc.T + row) * pc.D + d] * k[(kv_head * pc.S + col) * pc.D + d];
        }
        sum_exp += exp(dot * scale - max_score);
    }
    float acc = 0.0;
    for (uint col = 0u; col < pc.S; ++col) {
        float dot = 0.0;
        for (uint dd = 0u; dd < pc.D; ++dd) {
            dot += q[(head * pc.T + row) * pc.D + dd] * k[(kv_head * pc.S + col) * pc.D + dd];
        }
        float w = exp(dot * scale - max_score) / sum_exp;
        acc += w * v[(kv_head * pc.S + col) * pc.D + d_out];
    }
    output_values[(head * pc.T + row) * pc.D + d_out] = acc;
}
"""

_SOURCE_MASKED = """\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly QBuffer { float q[]; };
layout(set = 0, binding = 1) buffer restrict readonly KBuffer { float k[]; };
layout(set = 0, binding = 2) buffer restrict readonly VBuffer { float v[]; };
layout(set = 0, binding = 3) buffer restrict readonly MaskBuffer { float mask[]; };
layout(set = 0, binding = 4) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint NH; uint NK; uint T; uint S; uint D; } pc;
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint d_out = gl_WorkGroupID.z * 64u + gl_LocalInvocationID.x;
    if (head >= pc.NH || row >= pc.T || d_out >= pc.D) { return; }
    const uint kv_head = head * pc.NK / pc.NH;
    const float scale = inversesqrt(float(pc.D));
    float max_score = -1.0e38;
    for (uint col = 0u; col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[(head * pc.T + row) * pc.D + d] * k[(kv_head * pc.S + col) * pc.D + d];
        }
        float s = dot * scale + mask[row * pc.S + col];
        max_score = max(max_score, s);
    }
    float sum_exp = 0.0;
    for (uint col = 0u; col < pc.S; ++col) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[(head * pc.T + row) * pc.D + d] * k[(kv_head * pc.S + col) * pc.D + d];
        }
        sum_exp += exp(dot * scale + mask[row * pc.S + col] - max_score);
    }
    float acc = 0.0;
    for (uint col = 0u; col < pc.S; ++col) {
        float dot = 0.0;
        for (uint dd = 0u; dd < pc.D; ++dd) {
            dot += q[(head * pc.T + row) * pc.D + dd] * k[(kv_head * pc.S + col) * pc.D + dd];
        }
        float w = exp(dot * scale + mask[row * pc.S + col] - max_score) / sum_exp;
        acc += w * v[(kv_head * pc.S + col) * pc.D + d_out];
    }
    output_values[(head * pc.T + row) * pc.D + d_out] = acc;
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

    nh = q_shape[len(q_shape) - 3] if len(q_shape) >= 3 else 1
    nk = k_shape[len(k_shape) - 3] if len(k_shape) >= 3 else 1
    t = q_shape[len(q_shape) - 2] if len(q_shape) >= 2 else 1
    s = k_shape[len(k_shape) - 2] if len(k_shape) >= 2 else 1
    d = q_shape[len(q_shape) - 1]

    causal = _is_causal(node)
    masked = _has_mask(node)

    if causal:
        source = _SOURCE_CAUSAL
        shader_name = "export_sdpa_causal_f32"
    elif masked:
        source = _SOURCE_MASKED
        shader_name = "export_sdpa_masked_f32"
    else:
        source = _SOURCE_NONCAUSAL
        shader_name = "export_sdpa_f32"

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
                size=20,
                fields=(
                    PushConstantFieldSpec("NH", PushConstantType.UINT32, 0, nh),
                    PushConstantFieldSpec("NK", PushConstantType.UINT32, 4, nk),
                    PushConstantFieldSpec("T", PushConstantType.UINT32, 8, t),
                    PushConstantFieldSpec("S", PushConstantType.UINT32, 12, s),
                    PushConstantFieldSpec("D", PushConstantType.UINT32, 16, d),
                ),
            ),
            dispatch=(nh, t, (d + 63) // 64),
        ),
        source=source,
    )
