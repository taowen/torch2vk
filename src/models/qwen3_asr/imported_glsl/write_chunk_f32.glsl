#version 460

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict readonly XBuffer { float t_x[]; };
layout(set = 0, binding = 1) buffer restrict writeonly OutputBuffer { float t_output[]; };
layout(set = 0, binding = 2) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(push_constant) uniform PushConstants {
    int dst_step_offset;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint steps = uint(sizes.x);
    const uint hidden = uint(sizes.y);
    const uint total_steps = uint(sizes.z);
    const uint total = steps * hidden;
    if (index >= total) {
        return;
    }

    const uint step = index / hidden;
    const uint h = index % hidden;
    const uint dst_step = uint(pc.dst_step_offset) + step;
    if (dst_step >= total_steps) {
        return;
    }

    const uint dst_index = dst_step * hidden + h;
    t_output[dst_index] = t_x[index];
}
