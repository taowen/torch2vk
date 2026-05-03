#version 460

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { float t_output[]; };
layout(set = 0, binding = 1) buffer restrict readonly XBuffer { float t_x[]; };
layout(set = 0, binding = 2) buffer restrict readonly AlphaBuffer { float t_alpha[]; };
layout(set = 0, binding = 3) buffer restrict readonly WeightBuffer { float t_weight[]; };
layout(set = 0, binding = 4) buffer restrict readonly BiasBuffer { float t_bias[]; };
layout(set = 0, binding = 5) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

float snake(float v, float a) {
    const float aa = a + 1.0e-9;
    const float s = sin(a * v);
    return v + (s * s) / aa;
}

void main() {
    const uint oc = gl_GlobalInvocationID.x;
    const uint t = gl_GlobalInvocationID.y;
    const uint b = gl_GlobalInvocationID.z;
    const uint steps = uint(sizes.x);
    const uint batches = uint(sizes.y);
    const uint in_channels = uint(sizes.z);
    const uint out_channels = uint(sizes.w);
    if (oc >= out_channels || t >= steps || b >= batches) {
        return;
    }
    float acc = t_bias[oc];
    for (uint ic = 0u; ic < in_channels; ++ic) {
        const float a = t_alpha[ic];
        for (uint k = 0u; k < 7u; ++k) {
            const int ti = int(t) + int(k) * 3 - 9;
            if (ti < 0 || ti >= int(steps)) {
                continue;
            }
            const uint x_idx = (b * steps + uint(ti)) * in_channels + ic;
            const uint w_idx = (oc * in_channels + ic) * 7u + k;
            acc += snake(t_x[x_idx], a) * t_weight[w_idx];
        }
    }
    const uint out_idx = (b * steps + t) * out_channels + oc;
    t_output[out_idx] = acc;
}
