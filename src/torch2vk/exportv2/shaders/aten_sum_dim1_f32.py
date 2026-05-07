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

ATEN_SUM_DIM1_F32 = ShaderVariant(
    name="aten_sum_dim1_f32",
    family="aten",
    contract=ShaderContract(
        class_name="AtenSumDim1F32Program",
        shader_name="aten_sum_dim1_f32",
        fields=(
            TensorFieldSpec(
                "x", IOKind.INPUT, "x", TensorContract(dtype="float32", shape=("B", "C", "T", "H"))
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float32", shape=("B", "T", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("C", PushConstantType.UINT32, 0, "C"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 8, "H"),
            ),
        ),
        dispatch=(ceil_div("H", 256), "T", "B"),
    ),
    source="""
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint C; uint T; uint H; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
  const uint h = gl_GlobalInvocationID.x;
  const uint token = gl_GlobalInvocationID.y;
  const uint batch = gl_GlobalInvocationID.z;
  if (h >= pc.H || token >= pc.T) { return; }
  float acc = 0.0;
  for (uint codebook = 0u; codebook < pc.C; ++codebook) {
    const uint idx = ((batch * pc.C + codebook) * pc.T + token) * pc.H + h;
    acc += x[idx];
  }
  output_values[(batch * pc.T + token) * pc.H + h] = acc;
}
""".lstrip(),
)
