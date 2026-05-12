"""Quantization topology for generated Qwen3-ASR."""

from __future__ import annotations

from torch2vk.quantize import q4_k_m_more_bits_layer_indices


Q8_TENSOR_NAMES: tuple[str, ...] = ("thinker.model.embed_tokens.weight",)
Q8_TENSOR_PREFIXES: tuple[str, ...] = ("thinker.audio_tower.",)


def qwen3_asr_q4_k_m_q6_tensor_names(num_hidden_layers: int) -> tuple[str, ...]:
    layer_indices = q4_k_m_more_bits_layer_indices(num_hidden_layers)
    return (
        "thinker.lm_head.weight",
        *(f"thinker.model.layers.{idx}.self_attn.v_proj.weight" for idx in layer_indices),
        *(f"thinker.model.layers.{idx}.mlp.down_proj.weight" for idx in layer_indices),
    )
