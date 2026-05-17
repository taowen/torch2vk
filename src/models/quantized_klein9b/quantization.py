from __future__ import annotations

from torch2vk.quantize import Q4KMQuantizationConfig


def klein9b_q4_k_m_config() -> Q4KMQuantizationConfig:
    return Q4KMQuantizationConfig(
        model_name="FLUX.2-klein-9B",
        gguf_arch="flux2-klein9b",
        q8_tensor_prefixes=("",),
    )


def ae_q4_k_m_config() -> Q4KMQuantizationConfig:
    return Q4KMQuantizationConfig(
        model_name="FLUX.2 AutoEncoder",
        gguf_arch="flux2-ae",
        q8_tensor_prefixes=("",),
    )


def qwen3_text_encoder_q8_config() -> Q4KMQuantizationConfig:
    return Q4KMQuantizationConfig(
        model_name="Qwen3 text encoder",
        gguf_arch="qwen3",
        q8_tensor_prefixes=("model.",),
    )
