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


ROPE_TABLE_F32 = ShaderVariant(
    name="rope_table_f32",
    family="text",
    contract=ShaderContract(
        class_name="RopeTableF32Program",
        shader_name="rope_table_f32",
        fields=(
            TensorFieldSpec(
                name="start_position",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="int64", shape=(1,)),
            ),
            TensorFieldSpec(
                name="rope_theta",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=(1,)),
            ),
            TensorFieldSpec(
                name="cos",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=(1, "T", "D")),
            ),
            TensorFieldSpec(
                name="sin",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=(1, "T", "D")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec("T", PushConstantType.UINT32, 0, "T"),
                PushConstantFieldSpec("D", PushConstantType.UINT32, 4, "D"),
                PushConstantFieldSpec("attention_scaling", PushConstantType.FLOAT32, 8, 1.0),
                PushConstantFieldSpec("_reserved", PushConstantType.FLOAT32, 12, 0.0),
            ),
        ),
        dispatch=(ceil_div(mul("T", "D"), 256), 1, 1),
    ),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly StartPositionBuffer {
    int64_t start_position_values[];
};

layout(set = 0, binding = 1) buffer restrict readonly RopeThetaBuffer {
    float rope_theta_values[];
};

layout(set = 0, binding = 2) buffer restrict writeonly CosBuffer {
    float cos_values[];
};

layout(set = 0, binding = 3) buffer restrict writeonly SinBuffer {
    float sin_values[];
};

layout(push_constant) uniform PushConstants {
    uint T;
    uint D;
    float attention_scaling;
    float _reserved;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint total = pc.T * pc.D;
    if (index >= total) {
        return;
    }

    const uint token = index / pc.D;
    const uint d = index - token * pc.D;
    const uint half_dim = pc.D / 2u;
    const uint freq_idx = d % half_dim;
    const float exponent = (2.0 * float(freq_idx)) / float(pc.D);
    const float inv_freq = pow(rope_theta_values[0], -exponent);
    const float position = float(start_position_values[0] + int64_t(token));
    const float angle = position * inv_freq;
    cos_values[index] = cos(angle) * pc.attention_scaling;
    sin_values[index] = sin(angle) * pc.attention_scaling;
}
""".lstrip(),
)
