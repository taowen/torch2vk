"""Copied Agentorch shader modules used by the Qwen3 safetensor port."""

QWEN3_COPIED_SHADER_MODULES = (
    "add_f32_f32_f32_norepeat.py",
    "add_rms_f32_f32_f32_norepeat.py",
    "argmax_last_logits_f32.py",
    "argmax_last_logits_f32_parallel.py",
    "contig_cpy_f32_f16.py",
    "contig_cpy_f32_f32.py",
    "embedding_lookup_bf16_f32_sequence.py",
    "embedding_lookup_f16_f32_sequence.py",
    "fa_split_k_reduce.py",
    "flash_attn_f32_f16_aligned_f32accf16.py",
    "get_rows_f32_f32.py",
    "matmul_bf16_f32_f16acc_aligned_l.py",
    "matmul_f16_f32_f16acc_aligned_l.py",
    "mul_mat_vec_f16_f32_f32.py",
    "replace_sequence_span_f32.py",
    "rms_norm_f32_f32_weight_llama_wg512.py",
    "rms_norm_mul_partials_f32.py",
    "rms_norm_mul_rope_f32_f16.py",
    "rms_norm_mul_rope_f32_f32.py",
    "set_rows_f16_i64_token_major.py",
    "swiglu_f32.py",
)

