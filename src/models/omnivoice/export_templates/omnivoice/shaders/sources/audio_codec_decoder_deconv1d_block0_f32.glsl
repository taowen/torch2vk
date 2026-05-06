#version 460

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { float t_output[]; };
layout(set = 0, binding = 1) buffer restrict readonly XBuffer { float t_x[]; };
layout(set = 0, binding = 2) buffer restrict readonly WeightBuffer { float t_weight[]; };
layout(set = 0, binding = 3) buffer restrict readonly BiasBuffer { float t_bias[]; };
layout(set = 0, binding = 4) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint oc = gl_GlobalInvocationID.x;
    const uint to = gl_GlobalInvocationID.y;
    const uint b = gl_GlobalInvocationID.z;
    const uint si = uint(sizes.x);
    const uint so = uint(sizes.y);
    const uint ic = uint(sizes.z);
    const uint ocn = uint(sizes.w);
    if (oc >= ocn || to >= so) {
        return;
    }
    float acc = t_bias[oc];
    for (uint ii = 0u; ii < ic; ++ii) {
        for (uint k = 0u; k < 16u; ++k) {
            const int numer = int(to) + 4 - int(k);
            if ((numer & 7) != 0) {
                continue;
            }
            const int ti = numer / 8;
            if (ti < 0 || ti >= int(si)) {
                continue;
            }
            const uint x_idx = (b * si + uint(ti)) * ic + ii;
            const uint w_idx = (ii * ocn + oc) * 16u + k;
            acc += t_x[x_idx] * t_weight[w_idx];
        }
    }
    const uint out_idx = (b * so + to) * ocn + oc;
    t_output[out_idx] = acc;
}
