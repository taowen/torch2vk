"""Quantization topology for optimized Qwen3."""

from __future__ import annotations

from torch2vk.quantize import Q4KMQuantizationConfig, q4_k_m_more_bits_layer_indices


Q8_TENSOR_NAMES: tuple[str, ...] = ()
QUANTIZE_MODEL_NAME = "Qwen3"
QUANTIZE_GGUF_ARCH = "qwen3"


def qwen3_q4_k_m_uses_q6_layer(layer_idx: int, num_hidden_layers: int) -> bool:
    return layer_idx in q4_k_m_more_bits_layer_indices(num_hidden_layers)


def qwen3_q4_k_m_q6_tensor_names(num_hidden_layers: int) -> tuple[str, ...]:
    layer_indices = q4_k_m_more_bits_layer_indices(num_hidden_layers)
    return (
        "lm_head.weight",
        *(f"model.layers.{idx}.self_attn.v_proj.weight" for idx in layer_indices),
        *(f"model.layers.{idx}.mlp.down_proj.weight" for idx in layer_indices),
    )


def qwen3_q4_k_m_config(num_hidden_layers: int) -> Q4KMQuantizationConfig:
    return Q4KMQuantizationConfig(
        model_name=QUANTIZE_MODEL_NAME,
        gguf_arch=QUANTIZE_GGUF_ARCH,
        q6_tensor_names=qwen3_q4_k_m_q6_tensor_names(num_hidden_layers),
        q8_tensor_names=Q8_TENSOR_NAMES,
    )
