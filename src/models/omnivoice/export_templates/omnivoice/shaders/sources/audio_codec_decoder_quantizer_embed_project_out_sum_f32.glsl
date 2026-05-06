#version 460

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer { float t_output[]; };
layout(set = 0, binding = 1) buffer restrict readonly AudioIdsBuffer { int t_audio_ids[]; };
layout(set = 0, binding = 2) buffer restrict readonly Embed0Buffer { float t_embed0[]; };
layout(set = 0, binding = 3) buffer restrict readonly Embed1Buffer { float t_embed1[]; };
layout(set = 0, binding = 4) buffer restrict readonly Embed2Buffer { float t_embed2[]; };
layout(set = 0, binding = 5) buffer restrict readonly Embed3Buffer { float t_embed3[]; };
layout(set = 0, binding = 6) buffer restrict readonly Embed4Buffer { float t_embed4[]; };
layout(set = 0, binding = 7) buffer restrict readonly Embed5Buffer { float t_embed5[]; };
layout(set = 0, binding = 8) buffer restrict readonly Embed6Buffer { float t_embed6[]; };
layout(set = 0, binding = 9) buffer restrict readonly Embed7Buffer { float t_embed7[]; };
layout(set = 0, binding = 10) buffer restrict readonly Weight0Buffer { float t_weight0[]; };
layout(set = 0, binding = 11) buffer restrict readonly Bias0Buffer { float t_bias0[]; };
layout(set = 0, binding = 12) buffer restrict readonly Weight1Buffer { float t_weight1[]; };
layout(set = 0, binding = 13) buffer restrict readonly Bias1Buffer { float t_bias1[]; };
layout(set = 0, binding = 14) buffer restrict readonly Weight2Buffer { float t_weight2[]; };
layout(set = 0, binding = 15) buffer restrict readonly Bias2Buffer { float t_bias2[]; };
layout(set = 0, binding = 16) buffer restrict readonly Weight3Buffer { float t_weight3[]; };
layout(set = 0, binding = 17) buffer restrict readonly Bias3Buffer { float t_bias3[]; };
layout(set = 0, binding = 18) buffer restrict readonly Weight4Buffer { float t_weight4[]; };
layout(set = 0, binding = 19) buffer restrict readonly Bias4Buffer { float t_bias4[]; };
layout(set = 0, binding = 20) buffer restrict readonly Weight5Buffer { float t_weight5[]; };
layout(set = 0, binding = 21) buffer restrict readonly Bias5Buffer { float t_bias5[]; };
layout(set = 0, binding = 22) buffer restrict readonly Weight6Buffer { float t_weight6[]; };
layout(set = 0, binding = 23) buffer restrict readonly Bias6Buffer { float t_bias6[]; };
layout(set = 0, binding = 24) buffer restrict readonly Weight7Buffer { float t_weight7[]; };
layout(set = 0, binding = 25) buffer restrict readonly Bias7Buffer { float t_bias7[]; };
layout(set = 0, binding = 26) uniform restrict readonly sizes_UBO { ivec4 sizes; };

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint h = gl_GlobalInvocationID.x;
    const uint t = gl_GlobalInvocationID.y;
    const uint b = gl_GlobalInvocationID.z;
    const uint steps = uint(sizes.x);
    const uint batches = uint(sizes.y);
    if (h >= 1024u || t >= steps || b >= batches) {
        return;
    }
    const uint base = (b * 8u + 0u) * steps + t;
    const int id0 = t_audio_ids[base];
    const int id1 = t_audio_ids[base + steps];
    const int id2 = t_audio_ids[base + 2u * steps];
    const int id3 = t_audio_ids[base + 3u * steps];
    const int id4 = t_audio_ids[base + 4u * steps];
    const int id5 = t_audio_ids[base + 5u * steps];
    const int id6 = t_audio_ids[base + 6u * steps];
    const int id7 = t_audio_ids[base + 7u * steps];
    const uint w_base = h * 64u;
    const uint e0 = uint(id0) * 64u;
    const uint e1 = uint(id1) * 64u;
    const uint e2 = uint(id2) * 64u;
    const uint e3 = uint(id3) * 64u;
    const uint e4 = uint(id4) * 64u;
    const uint e5 = uint(id5) * 64u;
    const uint e6 = uint(id6) * 64u;
    const uint e7 = uint(id7) * 64u;
    float outv =
        t_bias0[h] + t_bias1[h] + t_bias2[h] + t_bias3[h] +
        t_bias4[h] + t_bias5[h] + t_bias6[h] + t_bias7[h];
    for (uint d = 0u; d < 64u; ++d) {
        const uint wi = w_base + d;
        outv += t_embed0[e0 + d] * t_weight0[wi];
        outv += t_embed1[e1 + d] * t_weight1[wi];
        outv += t_embed2[e2 + d] * t_weight2[wi];
        outv += t_embed3[e3 + d] * t_weight3[wi];
        outv += t_embed4[e4 + d] * t_weight4[wi];
        outv += t_embed5[e5 + d] * t_weight5[wi];
        outv += t_embed6[e6 + d] * t_weight6[wi];
        outv += t_embed7[e7 + d] * t_weight7[wi];
    }
    const uint out_idx = (b * steps + t) * 1024u + h;
    t_output[out_idx] = outv;
}
