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
)

TEXT_RMS_NORM_F32 = ShaderVariant(
    name="text_rms_norm_f32",
    family="text",
    contract=ShaderContract(
        class_name="TextRmsNormF32Program",
        shader_name="text_rms_norm_f32",
        fields=(
            TensorFieldSpec(
                "x", IOKind.INPUT, "input", TensorContract(dtype="float32", shape=(1, "T", "H"))
            ),
            TensorFieldSpec(
                "weight", IOKind.INPUT, "weight", TensorContract(dtype="bfloat16", shape=("H",))
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float32", shape=(1, "T", "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("T", PushConstantType.UINT32, 0, "T"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
                PushConstantFieldSpec("eps", PushConstantType.FLOAT32, 8, 1e-6),
            ),
        ),
        dispatch=("T", 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly WeightBuffer { uint16_t weight[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint T; uint H; float eps; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float partial_sumsq[256];
float bf16_to_f32(uint16_t value) { return uintBitsToFloat(uint(value) << 16); }
void main() {
  const uint row = gl_WorkGroupID.x;
  const uint tid = gl_LocalInvocationID.x;
  if (row >= pc.T) { return; }
  float sumsq = 0.0;
  for (uint h = tid; h < pc.H; h += 256u) { const float v = x[row * pc.H + h]; sumsq += v * v; }
  partial_sumsq[tid] = sumsq; barrier();
  for (uint stride = 128u; stride > 0u; stride >>= 1u) { if (tid < stride) { partial_sumsq[tid] += partial_sumsq[tid + stride]; } barrier(); }
  const float inv_rms = inversesqrt(partial_sumsq[0] / float(pc.H) + pc.eps);
  for (uint h = tid; h < pc.H; h += 256u) { output_values[row * pc.H + h] = x[row * pc.H + h] * inv_rms * bf16_to_f32(weight[h]); }
}
""".lstrip(),
)
