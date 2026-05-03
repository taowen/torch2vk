#version 460

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { float t_output[]; };
layout(set = 0, binding = 1) buffer restrict readonly XBuffer { float t_x[]; };
layout(set = 0, binding = 2) buffer restrict readonly ResidualBuffer { float t_residual[]; };
layout(set = 0, binding = 3) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint c = gl_GlobalInvocationID.x;
    const uint t = gl_GlobalInvocationID.y;
    const uint b = gl_GlobalInvocationID.z;
    const uint steps = uint(sizes.x);
    const uint batches = uint(sizes.y);
    const uint channels = uint(sizes.z);
    if (c >= channels || t >= steps || b >= batches) {
        return;
    }
    const uint idx = (b * steps + t) * channels + c;
    t_output[idx] = t_x[idx] + t_residual[idx];
}
