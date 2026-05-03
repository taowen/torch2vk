#version 460

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { float t_output[]; };
layout(set = 0, binding = 1) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint total = uint(t_output.length());
    if (index >= total) {
        return;
    }

    const uint hidden = uint(sizes.z);
    const uint half_channels = hidden / 2u;
    if (half_channels == 0u) {
        t_output[index] = 0.0;
        return;
    }

    const uint h = index % hidden;
    const uint s = (index / hidden) % uint(sizes.y);
    const float time = float(s + 1u);
    const float denom = float(max(1u, half_channels - 1u));
    const float log_increment = log(10000.0) / denom;

    if (h < half_channels) {
        const float freq = exp(-log_increment * float(h));
        t_output[index] = sin(time * freq);
        return;
    }

    const uint h2 = h - half_channels;
    const float freq = exp(-log_increment * float(h2));
    t_output[index] = cos(time * freq);
}
