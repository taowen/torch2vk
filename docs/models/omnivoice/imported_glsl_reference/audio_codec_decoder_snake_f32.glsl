#version 460

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { float t_output[]; };
layout(set = 0, binding = 1) buffer restrict readonly XBuffer { float t_x[]; };
layout(set = 0, binding = 2) buffer restrict readonly AlphaBuffer { float t_alpha[]; };
layout(set = 0, binding = 3) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint c = gl_GlobalInvocationID.x;
    const uint t = gl_GlobalInvocationID.y;
    const uint b = gl_GlobalInvocationID.z;
    const uint steps = uint(sizes.x);
    const uint batches = uint(sizes.y);
    const uint channels = uint(sizes.z);
    const uint alpha_len = uint(sizes.w);
    if (c >= channels || t >= steps || b >= batches) {
        return;
    }
    const uint idx = (b * steps + t) * channels + c;
    const float x = t_x[idx];
    const float a = t_alpha[c % alpha_len];
    const float aa = (abs(a) < 1.0e-8) ? 1.0e-8 : a;
    const float s = sin(aa * x);
    t_output[idx] = x + (s * s) / aa;
}
