"""Generated shader: omnivoice_input_embed_f32."""

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


OMNIVOICE_INPUT_EMBED_F32 = ShaderVariant(
    name='omnivoice_input_embed_f32',
    family='omnivoice',
    contract=ShaderContract(
        class_name='OmniVoiceInputEmbedF32Program',
        shader_name='omnivoice_input_embed_f32',
        fields=(
            TensorFieldSpec(
                name='text_weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='bfloat16', shape=('TV', 'H',)),
            ),
            TensorFieldSpec(
                name='audio_weight',
                io_kind=IOKind.INPUT,
                role='weight',
                contract=TensorContract(dtype='bfloat16', shape=('CV', 'H',)),
            ),
            TensorFieldSpec(
                name='batch_input_ids',
                io_kind=IOKind.INPUT,
                role='tokens',
                contract=TensorContract(dtype='int64', shape=('B', 'C', 'S',)),
            ),
            TensorFieldSpec(
                name='batch_audio_mask',
                io_kind=IOKind.INPUT,
                role='mask',
                contract=TensorContract(dtype='uint32', shape=('B', 'S',)),
            ),
            TensorFieldSpec(
                name='hidden_states',
                io_kind=IOKind.OUTPUT,
                role='hidden_states',
                contract=TensorContract(dtype='float32', shape=('B', 'S', 'H',)),
            ),
        ),
        push_constants=PushConstantSpec(
            size=20,
            fields=(
                PushConstantFieldSpec('B', PushConstantType.UINT32, 0, 'B', dynamic=False),
                PushConstantFieldSpec('C', PushConstantType.UINT32, 4, 'C', dynamic=False),
                PushConstantFieldSpec('S', PushConstantType.UINT32, 8, 'S', dynamic=False),
                PushConstantFieldSpec('H', PushConstantType.UINT32, 12, 'H', dynamic=False),
                PushConstantFieldSpec('V', PushConstantType.UINT32, 16, 1025, dynamic=False),
            ),
        ),
        params_buffer=None,
        dispatch=(ceil_div(mul(mul('B', 'S'), 'H'), 256), 1, 1),
    ),
    execution_requirements=ShaderExecutionRequirements(require_shader_int64=True),
    source="""\
#version 450

#extension GL_EXT_bfloat16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly TextWeightBuffer {
    bfloat16_t text_weight[];
};

layout(set = 0, binding = 1) buffer restrict readonly AudioWeightBuffer {
    bfloat16_t audio_weight[];
};

layout(set = 0, binding = 2) buffer restrict readonly BatchInputIdsBuffer {
    int64_t batch_input_ids[];
};

layout(set = 0, binding = 3) buffer restrict readonly BatchAudioMaskBuffer {
    uint batch_audio_mask[];
};

layout(set = 0, binding = 4) buffer restrict writeonly HiddenStatesBuffer {
    float hidden_states[];
};

layout(push_constant) uniform PushConstants {
    uint B;
    uint C;
    uint S;
    uint H;
    uint V;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    const uint total = pc.B * pc.S * pc.H;
    if (idx >= total) {
        return;
    }

    const uint h = idx % pc.H;
    const uint seq_idx = idx / pc.H;
    const uint s = seq_idx % pc.S;
    const uint b = seq_idx / pc.S;

    if (batch_audio_mask[b * pc.S + s] != 0u) {
        float value = 0.0;
        for (uint c = 0u; c < pc.C; ++c) {
            const uint input_offset = (b * pc.C + c) * pc.S + s;
            const uint token = uint(batch_input_ids[input_offset]);
            const uint row = c * pc.V + token;
            value += float(audio_weight[row * pc.H + h]);
        }
        hidden_states[idx] = value;
        return;
    }

    const uint text_offset = (b * pc.C) * pc.S + s;
    const uint text_token = uint(batch_input_ids[text_offset]);
    hidden_states[idx] = float(text_weight[text_token * pc.H + h]);
}
""",
)
