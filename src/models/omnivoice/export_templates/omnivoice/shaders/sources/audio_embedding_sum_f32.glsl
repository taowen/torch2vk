#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly OutputBuffer {
    float t_output[];
};

layout(set = 0, binding = 1) buffer restrict readonly AudioIdsBuffer {
    int t_audio_ids[];
};

layout(set = 0, binding = 2) buffer restrict readonly CodebookOffsetsBuffer {
    int t_codebook_offsets[];
};

layout(set = 0, binding = 3) buffer restrict readonly AudioEmbeddingsBuffer {
    float16_t t_audio_embeddings[];
};

layout(set = 0, binding = 4) uniform restrict readonly sizes_UBO {
    ivec4 sizes;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    const uint hidden_index = gl_GlobalInvocationID.x;
    const uint step_index = gl_GlobalInvocationID.y;
    const uint batch_index = gl_GlobalInvocationID.z;

    const uint hidden = uint(sizes.x);
    const uint steps = uint(sizes.y);
    const uint codebooks = uint(sizes.z);
    const uint vocab = uint(sizes.w);

    if (hidden_index >= hidden || step_index >= steps) {
        return;
    }
    // OmniVoice Stage0 fixed contract: exactly 8 codebooks.
    if (codebooks != 8u) {
        return;
    }

    const uint base = batch_index * codebooks * steps + step_index;
    const uint stride = steps;
    const uint id0 = uint(t_audio_ids[base + 0u * stride] + t_codebook_offsets[0]);
    const uint id1 = uint(t_audio_ids[base + 1u * stride] + t_codebook_offsets[1]);
    const uint id2 = uint(t_audio_ids[base + 2u * stride] + t_codebook_offsets[2]);
    const uint id3 = uint(t_audio_ids[base + 3u * stride] + t_codebook_offsets[3]);
    const uint id4 = uint(t_audio_ids[base + 4u * stride] + t_codebook_offsets[4]);
    const uint id5 = uint(t_audio_ids[base + 5u * stride] + t_codebook_offsets[5]);
    const uint id6 = uint(t_audio_ids[base + 6u * stride] + t_codebook_offsets[6]);
    const uint id7 = uint(t_audio_ids[base + 7u * stride] + t_codebook_offsets[7]);

    // Strict high-performance path: runtime guarantees ids are in range.
    float16_t value = float16_t(0.0);
    value += t_audio_embeddings[id0 * hidden + hidden_index];
    value += t_audio_embeddings[id1 * hidden + hidden_index];
    value += t_audio_embeddings[id2 * hidden + hidden_index];
    value += t_audio_embeddings[id3 * hidden + hidden_index];
    value += t_audio_embeddings[id4 * hidden + hidden_index];
    value += t_audio_embeddings[id5 * hidden + hidden_index];
    value += t_audio_embeddings[id6 * hidden + hidden_index];
    value += t_audio_embeddings[id7 * hidden + hidden_index];

    t_output[(batch_index * steps + step_index) * hidden + hidden_index] = float(value);
}
