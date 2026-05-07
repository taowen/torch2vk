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

ATEN_WHERE_F32 = ShaderVariant(
    name="aten_where_f32",
    family="aten",
    contract=ShaderContract(
        class_name="AtenWhereF32Program",
        shader_name="aten_where_f32",
        fields=(
            TensorFieldSpec(
                "mask", IOKind.INPUT, "mask", TensorContract(dtype="bool", shape=("B", "T", 1))
            ),
            TensorFieldSpec(
                "x", IOKind.INPUT, "x", TensorContract(dtype="float32", shape=("B", "T", "H"))
            ),
            TensorFieldSpec(
                "y", IOKind.INPUT, "y", TensorContract(dtype="float32", shape=("B", "T", "H"))
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float32", shape=("B", "T", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("T", PushConstantType.UINT32, 0, "T"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
            ),
        ),
        dispatch=(ceil_div("H", 256), "T", "B"),
    ),
    source="""
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly MaskBuffer { bool mask[]; };
layout(set = 0, binding = 1) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 2) buffer restrict readonly YBuffer { float y[]; };
layout(set = 0, binding = 3) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint T; uint H; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
  const uint h = gl_GlobalInvocationID.x;
  const uint token = gl_GlobalInvocationID.y;
  const uint batch = gl_GlobalInvocationID.z;
  if (h >= pc.H || token >= pc.T) { return; }
  const uint out_idx = (batch * pc.T + token) * pc.H + h;
  output_values[out_idx] = mask[batch * pc.T + token] ? x[out_idx] : y[out_idx];
}
""".lstrip(),
)
