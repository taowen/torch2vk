from __future__ import annotations

from torch2vk.runtime.shader import (
    IOKind,
    PushConstantFieldSpec,
    PushConstantInput,
    PushConstantSpec,
    PushConstantType,
    ShaderContract,
    ShaderVariant,
    TensorContract,
    TensorFieldSpec,
    add,
    ceil_div,
    mul,
)
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements


KLEIN9B_CAPTURE_QWEN3_CTX_F32 = ShaderVariant(
    name="klein9b_capture_qwen3_ctx_f32",
    family="klein9b",
    contract=ShaderContract(
        class_name="Klein9BCaptureQwen3CtxF32Program",
        shader_name="klein9b_capture_qwen3_ctx_f32",
        fields=(
            TensorFieldSpec(
                "layer_9",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "S", "H")),
            ),
            TensorFieldSpec(
                "layer_18",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "S", "H")),
            ),
            TensorFieldSpec(
                "layer_27",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float16", shape=("B", "S", "H")),
            ),
            TensorFieldSpec(
                "ctx",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float32", shape=("B", "S", mul(3, "H"))),
            ),
        ),
        push_constants=PushConstantSpec(
            size=12,
            fields=(
                PushConstantFieldSpec("S", PushConstantType.UINT32, 0, "S"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 4, "H"),
                PushConstantFieldSpec("N", PushConstantType.UINT32, 8, mul(mul("B", "S"), "H")),
            ),
        ),
        dispatch=(ceil_div(mul(mul("B", "S"), "H"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly Layer9Buffer { float16_t layer_9[]; };
layout(set = 0, binding = 1) buffer restrict readonly Layer18Buffer { float16_t layer_18[]; };
layout(set = 0, binding = 2) buffer restrict readonly Layer27Buffer { float16_t layer_27[]; };
layout(set = 0, binding = 3) buffer restrict writeonly CtxBuffer { float ctx[]; };
layout(push_constant) uniform PushConstants { uint S; uint H; uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const uint hidden = idx % pc.H;
    const uint token = (idx / pc.H) % pc.S;
    const uint batch = idx / (pc.H * pc.S);
    const uint out_base = (batch * pc.S + token) * (3u * pc.H) + hidden;
    ctx[out_base] = float(layer_9[idx]);
    ctx[out_base + pc.H] = float(layer_18[idx]);
    ctx[out_base + 2u * pc.H] = float(layer_27[idx]);
}
""",
)


KLEIN9B_EULER_UPDATE_F32 = ShaderVariant(
    name="klein9b_euler_update_f32",
    family="klein9b",
    contract=ShaderContract(
        class_name="Klein9BEulerUpdateF32Program",
        shader_name="klein9b_euler_update_f32",
        fields=(
            TensorFieldSpec(
                "x",
                IOKind.INOUT,
                "state",
                TensorContract(dtype="float32", shape=("B", "S", "C")),
            ),
            TensorFieldSpec(
                "pred",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float32", shape=("B", "S", "C")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=8,
            fields=(
                PushConstantFieldSpec("N", PushConstantType.UINT32, 0, mul(mul("B", "S"), "C")),
                PushConstantFieldSpec("dt", PushConstantType.FLOAT32, 4, PushConstantInput("dt")),
            ),
        ),
        dispatch=(ceil_div(mul(mul("B", "S"), "C"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict readonly PredBuffer { float pred[]; };
layout(push_constant) uniform PushConstants { uint N; float dt; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    x[idx] = x[idx] + pc.dt * pred[idx];
}
""",
)


KLEIN9B_CAT_TXT_IMG_F32 = ShaderVariant(
    name="klein9b_cat_txt_img_f32",
    family="klein9b",
    contract=ShaderContract(
        class_name="Klein9BCatTxtImgF32Program",
        shader_name="klein9b_cat_txt_img_f32",
        fields=(
            TensorFieldSpec(
                "txt",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float32", shape=("B", "T", "H")),
            ),
            TensorFieldSpec(
                "img",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float32", shape=("B", "S", "H")),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float32", shape=("B", add("T", "S"), "H")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec("N", PushConstantType.UINT32, 0, mul(mul("B", add("T", "S")), "H")),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 8, "S"),
                PushConstantFieldSpec("H", PushConstantType.UINT32, 12, "H"),
            ),
        ),
        dispatch=(ceil_div(mul(mul("B", add("T", "S")), "H"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly TxtBuffer { float txt[]; };
layout(set = 0, binding = 1) buffer restrict readonly ImgBuffer { float img[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint T; uint S; uint H; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const uint h = idx % pc.H;
    const uint token = (idx / pc.H) % (pc.T + pc.S);
    const uint batch = idx / (pc.H * (pc.T + pc.S));
    if (token < pc.T) {
        output_values[idx] = txt[(batch * pc.T + token) * pc.H + h];
    } else {
        const uint img_token = token - pc.T;
        output_values[idx] = img[(batch * pc.S + img_token) * pc.H + h];
    }
}
""",
)


KLEIN9B_CAT_PE_F32 = ShaderVariant(
    name="klein9b_cat_pe_f32",
    family="klein9b",
    contract=ShaderContract(
        class_name="Klein9BCatPeF32Program",
        shader_name="klein9b_cat_pe_f32",
        fields=(
            TensorFieldSpec(
                "pe_ctx",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float32", shape=("B", "A", "T", "D", "R", "C")),
            ),
            TensorFieldSpec(
                "pe_x",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float32", shape=("B", "A", "S", "D", "R", "C")),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float32", shape=("B", "A", add("T", "S"), "D", "R", "C")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=16,
            fields=(
                PushConstantFieldSpec(
                    "N",
                    PushConstantType.UINT32,
                    0,
                    mul(mul(mul(mul(mul("B", "A"), add("T", "S")), "D"), "R"), "C"),
                ),
                PushConstantFieldSpec("T", PushConstantType.UINT32, 4, "T"),
                PushConstantFieldSpec("S", PushConstantType.UINT32, 8, "S"),
                PushConstantFieldSpec("INNER", PushConstantType.UINT32, 12, mul(mul("D", "R"), "C")),
            ),
        ),
        dispatch=(ceil_div(mul(mul(mul(mul(mul("B", "A"), add("T", "S")), "D"), "R"), "C"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(),
    source="""\
#version 450
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly PeCtxBuffer { float pe_ctx[]; };
layout(set = 0, binding = 1) buffer restrict readonly PeXBuffer { float pe_x[]; };
layout(set = 0, binding = 2) buffer restrict writeonly OutputBuffer { float output_values[]; };
layout(push_constant) uniform PushConstants { uint N; uint T; uint S; uint INNER; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    const uint inner = idx % pc.INNER;
    const uint token = (idx / pc.INNER) % (pc.T + pc.S);
    const uint outer = idx / (pc.INNER * (pc.T + pc.S));
    if (token < pc.T) {
        output_values[idx] = pe_ctx[(outer * pc.T + token) * pc.INNER + inner];
    } else {
        const uint img_token = token - pc.T;
        output_values[idx] = pe_x[(outer * pc.S + img_token) * pc.INNER + inner];
    }
}
""",
)


KLEIN9B_LATENT_F32_TO_F16 = ShaderVariant(
    name="klein9b_latent_f32_to_f16",
    family="klein9b",
    contract=ShaderContract(
        class_name="Klein9BLatentF32ToF16Program",
        shader_name="klein9b_latent_f32_to_f16",
        fields=(
            TensorFieldSpec(
                "x",
                IOKind.INPUT,
                "input",
                TensorContract(dtype="float32", shape=("B", "S", "C")),
            ),
            TensorFieldSpec(
                "output",
                IOKind.OUTPUT,
                "output",
                TensorContract(dtype="float16", shape=("B", "H", "W", "C")),
            ),
        ),
        push_constants=PushConstantSpec(
            size=4,
            fields=(
                PushConstantFieldSpec("N", PushConstantType.UINT32, 0, mul(mul(mul("B", "H"), "W"), "C")),
            ),
        ),
        dispatch=(ceil_div(mul(mul(mul("B", "H"), "W"), "C"), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_storage_buffer_16bit_access=True),
    source="""\
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float16_t output_values[]; };
layout(push_constant) uniform PushConstants { uint N; } pc;
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.N) { return; }
    output_values[idx] = float16_t(x[idx]);
}
""",
)
