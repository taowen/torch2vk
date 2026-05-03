#version 460

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly OutputIdsBuffer {
    int t_output_ids[];
};

layout(set = 0, binding = 1) buffer restrict readonly LogitsBuffer {
    float t_logits[];
};

layout(set = 0, binding = 2) buffer restrict readonly CodebookOffsetsBuffer {
    int t_codebook_offsets[];
};

layout(set = 0, binding = 3) uniform restrict readonly sizes_UBO {
    ivec4 sizes;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
shared float best_vals[256];
shared int best_ids[256];

void main() {
    const uint codebook = gl_WorkGroupID.x;
    const uint step_index = gl_WorkGroupID.y;
    const uint batch_index = gl_WorkGroupID.z;
    const uint lane = gl_LocalInvocationID.x;

    const uint steps = uint(sizes.x);
    const uint batches = uint(sizes.y);
    const uint vocab = uint(sizes.z);
    const uint codebooks = uint(sizes.w);
    if (step_index >= steps || batch_index >= batches || codebooks != 8u || codebook >= codebooks) {
        return;
    }

    const uint logits_base = (batch_index * steps + step_index) * vocab;
    const uint out_base = (batch_index * codebooks) * steps + step_index;
    const int start = t_codebook_offsets[codebook];
    const int end = (codebook + 1u < codebooks) ? t_codebook_offsets[codebook + 1u] : int(vocab);
    const int seg_len = max(0, end - start);

    float local_best_val = -3.402823466e+38;
    int local_best_id = 0;
    for (int i = int(lane); i < seg_len; i += 256) {
        const float v = t_logits[logits_base + uint(start + i)];
        if (v > local_best_val) {
            local_best_val = v;
            local_best_id = i;
        }
    }
    best_vals[lane] = local_best_val;
    best_ids[lane] = local_best_id;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            const float rhs_val = best_vals[lane + stride];
            if (rhs_val > best_vals[lane]) {
                best_vals[lane] = rhs_val;
                best_ids[lane] = best_ids[lane + stride];
            }
        }
        barrier();
    }
    if (lane == 0u) {
        t_output_ids[out_base + codebook * steps] = best_ids[0];
    }
}
