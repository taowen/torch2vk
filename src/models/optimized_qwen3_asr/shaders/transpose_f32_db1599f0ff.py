"""Generated shader: transpose_f32_db1599f0ff."""

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
from torch2vk.vulkan.shader_execution_requirements import (
    ShaderExecutionRequirements,
)


TRANSPOSE_F32_DB1599F0FF = ShaderVariant(
    name="transpose_f32_db1599f0ff",
    family="export",
    contract=ShaderContract(
        class_name="ExportTransposeProgram",
        shader_name="transpose_f32_db1599f0ff",
        fields=(
            TensorFieldSpec(
                name="x",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(
                    dtype="float16",
                    shape=(
                        "I0",
                        "I1",
                        "I2",
                        "I3",
                    ),
                ),
            ),
            TensorFieldSpec(
                name="output",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(
                    dtype="float16",
                    shape=(
                        "O0",
                        "O1",
                        "O2",
                        "O3",
                    ),
                ),
            ),
        ),
        push_constants=PushConstantSpec(
            size=36,
            fields=(
                PushConstantFieldSpec(
                    "N",
                    PushConstantType.UINT32,
                    0,
                    mul(mul(mul("O0", "O1"), "O2"), "O3"),
                    dynamic=False,
                ),
                PushConstantFieldSpec("O0", PushConstantType.UINT32, 4, "O0", dynamic=False),
                PushConstantFieldSpec("O1", PushConstantType.UINT32, 8, "O1", dynamic=False),
                PushConstantFieldSpec("O2", PushConstantType.UINT32, 12, "O2", dynamic=False),
                PushConstantFieldSpec("O3", PushConstantType.UINT32, 16, "O3", dynamic=False),
                PushConstantFieldSpec("I0", PushConstantType.UINT32, 20, "I0", dynamic=False),
                PushConstantFieldSpec("I1", PushConstantType.UINT32, 24, "I1", dynamic=False),
                PushConstantFieldSpec("I2", PushConstantType.UINT32, 28, "I2", dynamic=False),
                PushConstantFieldSpec("I3", PushConstantType.UINT32, 32, "I3", dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul(mul("O0", "O1"), "O2"), "O3"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float16_t x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants {
    uint N;
    uint O0;
    uint O1;
    uint O2;
    uint O3;
    uint I0;
    uint I1;
    uint I2;
    uint I3;
} pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.N) {
        uint rem = idx;
        uint c3 = rem % pc.O3;
        rem = rem / pc.O3;
        uint c2 = rem % pc.O2;
        rem = rem / pc.O2;
        uint c1 = rem % pc.O1;
        rem = rem / pc.O1;
        uint c0 = rem % pc.O0;
        rem = rem / pc.O0;
        uint in_idx = 0u;
        in_idx = in_idx * pc.I0 + c0;
        in_idx = in_idx * pc.I1 + c2;
        in_idx = in_idx * pc.I2 + c1;
        in_idx = in_idx * pc.I3 + c3;
        output_values[idx] = x[in_idx];
    }
}
""",
)
