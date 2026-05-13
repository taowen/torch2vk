"""Quantization topology for generated Qwen3-ASR."""

from __future__ import annotations

from torch2vk.quantize import Q4KMQuantizationConfig, q4_k_m_more_bits_layer_indices


Q8_TENSOR_NAMES: tuple[str, ...] = ("thinker.model.embed_tokens.weight",)
Q8_TENSOR_PREFIXES: tuple[str, ...] = ("thinker.audio_tower.",)
QUANTIZE_MODEL_NAME = "Qwen3-ASR"
QUANTIZE_GGUF_ARCH = "qwen3-asr"


def qwen3_asr_q4_k_m_q6_tensor_names(num_hidden_layers: int) -> tuple[str, ...]:
    layer_indices = q4_k_m_more_bits_layer_indices(num_hidden_layers)
    return (
        "thinker.lm_head.weight",
        *(f"thinker.model.layers.{idx}.self_attn.v_proj.weight" for idx in layer_indices),
        *(f"thinker.model.layers.{idx}.mlp.down_proj.weight" for idx in layer_indices),
    )


def qwen3_asr_q4_k_m_config(num_hidden_layers: int) -> Q4KMQuantizationConfig:
    return Q4KMQuantizationConfig(
        model_name=QUANTIZE_MODEL_NAME,
        gguf_arch=QUANTIZE_GGUF_ARCH,
        q6_tensor_names=qwen3_asr_q4_k_m_q6_tensor_names(num_hidden_layers),
        q8_tensor_names=Q8_TENSOR_NAMES,
        q8_tensor_prefixes=Q8_TENSOR_PREFIXES,
    )
