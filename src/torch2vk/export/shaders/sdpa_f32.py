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
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
shared float scores[1024];
void main() {
    const uint head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint tid = gl_LocalInvocationID.x;
    if (head >= pc.NH || row >= pc.T) { return; }
    const uint kv_head = head * pc.NK / pc.NH;
    const float scale = inversesqrt(float(pc.D));
    for (uint col = tid; col < pc.S; col += 128u) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[(head * pc.T + row) * pc.D + d] * k[(kv_head * pc.S + col) * pc.D + d];
        }
        scores[col] = (col <= row) ? dot * scale : -1.0e38;
    }
    barrier();
    float max_score = -1.0e38;
    for (uint col = 0u; col < pc.S; ++col) { max_score = max(max_score, scores[col]); }
    float sum_exp = 0.0;
    for (uint col = 0u; col < pc.S; ++col) {
        scores[col] = exp(scores[col] - max_score);
        sum_exp += scores[col];
    }
    for (uint col = 0u; col < pc.S; ++col) { scores[col] /= sum_exp; }
    barrier();
    for (uint d = tid; d < pc.D; d += 128u) {
        float acc = 0.0;
        for (uint col = 0u; col < pc.S; ++col) {
            acc += scores[col] * v[(kv_head * pc.S + col) * pc.D + d];
        }
        output_values[(head * pc.T + row) * pc.D + d] = acc;
    }
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
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
shared float scores[1024];
void main() {
    const uint head = gl_WorkGroupID.x;
    const uint row = gl_WorkGroupID.y;
    const uint tid = gl_LocalInvocationID.x;
    if (head >= pc.NH || row >= pc.T) { return; }
    const uint kv_head = head * pc.NK / pc.NH;
    const float scale = inversesqrt(float(pc.D));
    for (uint col = tid; col < pc.S; col += 128u) {
        float dot = 0.0;
        for (uint d = 0u; d < pc.D; ++d) {
            dot += q[(head * pc.T + row) * pc.D + d] * k[(kv_head * pc.S + col) * pc.D + d];
        }
        scores[col] = dot * scale;
    }
    barrier();
    float max_score = -1.0e38;
    for (uint col = 0u; col < pc.S; ++col) { max_score = max(max_score, scores[col]); }
    float sum_exp = 0.0;
    for (uint col = 0u; col < pc.S; ++col) {
        scores[col] = exp(scores[col] - max_score);
        sum_exp += scores[col];
    }
    for (uint col = 0u; col < pc.S; ++col) { scores[col] /= sum_exp; }
    barrier();
    for (uint d = tid; d < pc.D; d += 128u) {
        float acc = 0.0;
        for (uint col = 0u; col < pc.S; ++col) {
            acc += scores[col] * v[(kv_head * pc.S + col) * pc.D + d];
        }
        output_values[(head * pc.T + row) * pc.D + d] = acc;
    }
}
"""


def _is_causal(node: Node) -> bool:
    if len(node.args) >= 6 and isinstance(node.args[5], bool):
        return node.args[5]
    return False


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
    source = _SOURCE_CAUSAL if causal else _SOURCE_NONCAUSAL
    shader_name = "export_sdpa_causal_f32" if causal else "export_sdpa_f32"

    return ShaderVariant(
        name=shader_name,
        family="export",
        contract=ShaderContract(
            class_name="ExportSdpaProgram",
            shader_name=shader_name,
            fields=(
                TensorFieldSpec("q", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=q_contract)),
                TensorFieldSpec("k", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=k_contract)),
                TensorFieldSpec("v", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=v_contract)),
                TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="float32", shape=out_contract)),
            ),
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
            dispatch=(nh, t, 1),
        ),
        source=source,
    )
