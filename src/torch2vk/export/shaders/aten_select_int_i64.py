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
)

ATEN_SELECT_INT_I64 = ShaderVariant(
    name="aten_select_int_i64",
    family="aten",
    contract=ShaderContract(
        class_name="AtenSelectIntI64Program",
        shader_name="aten_select_int_i64",
        fields=(
            TensorFieldSpec("x", IOKind.INPUT, "x", TensorContract(dtype="int64", shape=("B", "C", "T"))),
            TensorFieldSpec("output", IOKind.OUTPUT, "output", TensorContract(dtype="int64", shape=("B", "T"))),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("C", PushConstantType.UINT32, 0, "C"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
            ),
        ),
        dispatch=(ceil_div("T", 256), "B", 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly InputBuffer { int64_t x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { int64_t output_values[]; };
layout(push_constant) uniform PushConstants { uint C; uint T; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
  const uint token = gl_GlobalInvocationID.x;
  const uint batch = gl_GlobalInvocationID.y;
  if (token >= pc.T) { return; }
  output_values[batch * pc.T + token] = x[(batch * pc.C) * pc.T + token];
}
""".lstrip(),
)
