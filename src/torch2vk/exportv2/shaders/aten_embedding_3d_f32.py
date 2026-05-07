from __future__ import annotations

from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderExecutionRequirements,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
    ceil_div,
    mul,
)

ATEN_EMBEDDING_3D_F32 = ShaderVariant(
    name="aten_embedding_3d_f32",
    family="aten",
    contract=ShaderContract(
        class_name="AtenEmbedding3DF32Program",
        shader_name="aten_embedding_3d_f32",
        fields=(
            TensorFieldSpec(
                "weight", IOKind.INPUT, "weight", TensorContract(dtype="float32", shape=("V", "H"))
            ),
            TensorFieldSpec(
                "indices",
                IOKind.INPUT,
                "indices",
                TensorContract(dtype="int64", shape=("B", "C", "T")),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float32", shape=("B", "C", "T", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec("C", PushConstantType.UINT32, 0, "C"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 8, "H"),
                PushConstantFieldSpec("V", PushConstantType.UINT32, 12, "V"),
            ),
        ),
        dispatch=(ceil_div("H", 256), "T", mul("B", "C")),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly WeightBuffer { float weight[]; };
layout(set = 0, binding = 1) buffer restrict readonly IndicesBuffer { int64_t indices[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint C; uint T; uint H; uint V; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
  const uint h = gl_GlobalInvocationID.x;
  const uint token = gl_GlobalInvocationID.y;
  const uint bc = gl_GlobalInvocationID.z;
  if (h >= pc.H || token >= pc.T) { return; }
  const uint batch = bc / pc.C;
  const uint codebook = bc % pc.C;
  const uint idx = (batch * pc.C + codebook) * pc.T + token;
  const int64_t token_id = indices[idx];
  const uint out_idx = idx * pc.H + h;
  output_values[out_idx] = (token_id >= 0 && token_id < int64_t(pc.V)) ? weight[uint(token_id) * pc.H + h] : 0.0;
}
""".lstrip(),
)
