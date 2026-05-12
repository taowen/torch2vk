"""Reusable RoPE cos/sin table generation."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.runtime.logical import LogicalTensor, MemoryClass, TensorLifetime, TensorRole
from torch2vk.runtime.session import RuntimeSession
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
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements
from torch2vk.vulkan.types import TensorSpec


@dataclass(frozen=True, slots=True)
class RopeTableTensors:
    start_position: LogicalTensor
    theta: LogicalTensor
    cos: LogicalTensor
    sin: LogicalTensor


ROPE_TABLE_F32 = ShaderVariant(
    name="rope_table_f32",
    family="torch2vk.rope",
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
                name="theta",
                io_kind=IOKind.INPUT,
                role="input",
                contract=TensorContract(dtype="float32", shape=(1,)),
            ),
            TensorFieldSpec(
                name="cos",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float16", shape=("B", "T", "D")),
            ),
            TensorFieldSpec(
                name="sin",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float16", shape=("B", "T", "D")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=20,
            fields=(
                PushConstantFieldSpec("B", PushConstantType.UINT32, 0, "B"),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
                PushConstantFieldSpec("D", PushConstantType.UINT32, 8, "D"),
                PushConstantFieldSpec("attention_scaling", PushConstantType.FLOAT32, 12, 1.0),
                PushConstantFieldSpec("_reserved", PushConstantType.FLOAT32, 16, 0.0),
            ),
        ),
        dispatch=(ceil_div(mul(mul("B", "T"), "D"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(
        require_shader_int64=True,
        require_storage_buffer_16bit_access=True,
    ),
    source="""
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly StartPositionBuffer {
    int64_t start_position_values[];
};

layout(set = 0, binding = 1) buffer restrict readonly ThetaBuffer {
    float theta_values[];
};

layout(set = 0, binding = 2) buffer restrict writeonly CosBuffer {
    float16_t cos_values[];
};

layout(set = 0, binding = 3) buffer restrict writeonly SinBuffer {
    float16_t sin_values[];
};

layout(push_constant) uniform PushConstants {
    uint B;
    uint T;
    uint D;
    float attention_scaling;
    float _reserved;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.T * pc.D;
    if (index >= total) {
        return;
    }

    const uint token_dim = index % (pc.T * pc.D);
    const uint token = token_dim / pc.D;
    const uint d = token_dim - token * pc.D;
    const uint half_dim = pc.D / 2u;
    const uint freq_idx = d % half_dim;
    const float exponent = (2.0 * float(freq_idx)) / float(pc.D);
    const float inv_freq = pow(theta_values[0], -exponent);
    const float position = float(start_position_values[0] + int64_t(token));
    const float angle = position * inv_freq;
    cos_values[index] = float16_t(cos(angle) * pc.attention_scaling);
    sin_values[index] = float16_t(sin(angle) * pc.attention_scaling);
}
""".lstrip(),
)


ROPE_TABLE_OUTPUT_F32 = ShaderVariant(
    name="rope_table_output_f32",
    family=ROPE_TABLE_F32.family,
    contract=ShaderContract(
        class_name="RopeTableOutputF32Program",
        shader_name="rope_table_output_f32",
        fields=(
            ROPE_TABLE_F32.contract.fields[0],
            ROPE_TABLE_F32.contract.fields[1],
            TensorFieldSpec(
                name="cos",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=("B", "T", "D")),
            ),
            TensorFieldSpec(
                name="sin",
                io_kind=IOKind.OUTPUT,
                role="output",
                contract=TensorContract(dtype="float32", shape=("B", "T", "D")),
            ),
        ),
        push_constants=ROPE_TABLE_F32.contract.push_constants,
        dispatch=ROPE_TABLE_F32.contract.dispatch,
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source=ROPE_TABLE_F32.source.replace(
        "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
        "#extension GL_EXT_shader_16bit_storage : require\n",
        "",
    )
    .replace("float16_t cos_values[];", "float cos_values[];")
    .replace("float16_t sin_values[];", "float sin_values[];")
    .replace(
        "float16_t(cos(angle) * pc.attention_scaling)",
        "cos(angle) * pc.attention_scaling",
    )
    .replace(
        "float16_t(sin(angle) * pc.attention_scaling)",
        "sin(angle) * pc.attention_scaling",
    ),
)


def declare_rope_start_position_tensor(name: str) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype="int64", shape=(1,)),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )


def declare_rope_theta_tensor(name: str) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype="float32", shape=(1,)),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )


def declare_rope_table_tensors(
    prefix: str,
    *,
    batch: int,
    sequence_length: int,
    head_dim: int,
    dtype: str = "float16",
) -> RopeTableTensors:
    return RopeTableTensors(
        start_position=declare_rope_start_position_tensor(f"{prefix}.start_position"),
        theta=declare_rope_theta_tensor(f"{prefix}.theta"),
        cos=_declare_rope_output(f"{prefix}.cos", batch=batch, sequence_length=sequence_length, head_dim=head_dim, dtype=dtype),
        sin=_declare_rope_output(f"{prefix}.sin", batch=batch, sequence_length=sequence_length, head_dim=head_dim, dtype=dtype),
    )


def run_rope_table_f32(
    rt: RuntimeSession,
    *,
    start_position: LogicalTensor,
    theta: LogicalTensor,
    cos: LogicalTensor,
    sin: LogicalTensor,
    frame_name: str,
) -> None:
    if cos.spec.dtype == "float32":
        variant = ROPE_TABLE_OUTPUT_F32
    elif cos.spec.dtype == "float16":
        variant = ROPE_TABLE_F32
    else:
        raise ValueError(f"unsupported RoPE table dtype: {cos.spec.dtype}")
    with rt.frame(frame_name):
        variant(
            rt,
            start_position=start_position,
            theta=theta,
            cos=cos,
            sin=sin,
        )


def _declare_rope_output(
    name: str,
    *,
    batch: int,
    sequence_length: int,
    head_dim: int,
    dtype: str,
) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=(batch, sequence_length, head_dim)),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )
