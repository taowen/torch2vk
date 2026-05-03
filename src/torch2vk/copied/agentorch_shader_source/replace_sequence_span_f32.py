"""Embedded GLSL for replacing a contiguous sequence span in float32 activations."""

from __future__ import annotations

from agentorch.kernel.contract import (
    input_tensor,
    output_tensor,
    shader_contract,
    storage_buffer_binding,
    uniform_buffer_binding,
    uniform_ivec4,
)

from .shader_variant import shader_variant


REPLACE_SEQUENCE_SPAN_F32 = shader_variant(
    name="replace_sequence_span_f32",
    family="replace_sequence_span_f32",
    contract=shader_contract(
        class_name="ReplaceSequenceSpanF32Program",
        shader_name="replace_sequence_span_f32",
        fields=(
            input_tensor(name="x", binding="t_input", role="x", dtypes=("float32",), shape=("B", "S", "H")),
            input_tensor(
                name="replacement",
                binding="t_replacement",
                role="replacement",
                dtypes=("float32",),
                shape=("B", "R", "H"),
            ),
            input_tensor(name="span_start", binding="t_span_start", role="span_start", dtypes=("int32",), shape=("B",)),
            output_tensor(name="output", binding="t_output", role="output", dtypes=("float32",), shape=("B", "S", "H")),
        ),
        uniforms=(
            uniform_ivec4(binding="sizes", value=("H", "S", "B", 1)),
            uniform_ivec4(binding="replacement_sizes", value=("H", "R", "B", 1)),
        ),
        dispatch=("H", "S", "B"),
        bindings=(
            storage_buffer_binding(name="t_output", binding=0),
            storage_buffer_binding(name="t_input", binding=1),
            storage_buffer_binding(name="t_replacement", binding=2),
            storage_buffer_binding(name="t_span_start", binding=3),
            uniform_buffer_binding(name="sizes", binding=4),
            uniform_buffer_binding(name="replacement_sizes", binding=5),
        ),
    ),
    source="""
#version 460

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { float t_output[]; };
layout(set = 0, binding = 1) buffer restrict readonly InputBuffer { float t_input[]; };
layout(set = 0, binding = 2) buffer restrict readonly ReplacementBuffer { float t_replacement[]; };
layout(set = 0, binding = 3) buffer restrict readonly SpanStartBuffer { int t_span_start[]; };
layout(set = 0, binding = 4) uniform restrict readonly sizes_UBO { ivec4 sizes; };
layout(set = 0, binding = 5) uniform restrict readonly replacement_sizes_UBO { ivec4 replacement_sizes; };

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uint offset3(const uint b, const uint s, const uint h, const uint steps, const uint hidden) {
    return ((b * steps + s) * hidden) + h;
}

void main() {
    const uint h = gl_GlobalInvocationID.x;
    const uint s = gl_GlobalInvocationID.y;
    const uint b = gl_GlobalInvocationID.z;
    const uint hidden = uint(sizes.x);
    const uint steps = uint(sizes.y);
    const uint batches = uint(sizes.z);
    const uint replacement_steps = uint(replacement_sizes.y);
    if (h >= hidden || s >= steps || b >= batches) {
        return;
    }
    const int replacement_index = int(s) - t_span_start[b];
    const uint output_index = offset3(b, s, h, steps, hidden);
    if (replacement_index >= 0 && uint(replacement_index) < replacement_steps) {
        t_output[output_index] = t_replacement[offset3(b, uint(replacement_index), h, replacement_steps, hidden)];
    } else {
        t_output[output_index] = t_input[output_index];
    }
}
""".lstrip(),
)
