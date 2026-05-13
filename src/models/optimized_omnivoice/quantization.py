"""Quantization policy for optimized OmniVoice."""

from __future__ import annotations

from torch2vk.quantize import Q4KMQuantizationConfig, q4_k_m_more_bits_layer_indices


QUANTIZE_MODEL_NAME = "OmniVoice"
QUANTIZE_GGUF_ARCH = "clip"

Q8_TENSOR_NAMES: tuple[str, ...] = (
    "llm.embed_tokens.weight",
    "audio_embeddings.weight",
    "acoustic_decoder.conv1.weight",
    "acoustic_decoder.conv2.weight",
    *(f"acoustic_decoder.block.{block_idx}.conv_t1.weight" for block_idx in range(5)),
    *(
        f"acoustic_decoder.block.{block_idx}.res_unit{unit_idx}.conv{conv_idx}.weight"
        for block_idx in range(5)
        for unit_idx in range(1, 4)
        for conv_idx in range(1, 3)
    ),
)

Q8_TENSOR_PREFIXES: tuple[str, ...] = (
    "quantizer.",
    "fc2.",
)


def omnivoice_q4_k_m_q6_tensor_names(num_hidden_layers: int) -> tuple[str, ...]:
    layer_indices = q4_k_m_more_bits_layer_indices(num_hidden_layers)
    return (
        "audio_heads.weight",
        *(f"llm.layers.{idx}.self_attn.v_proj.weight" for idx in layer_indices),
        *(f"llm.layers.{idx}.mlp.down_proj.weight" for idx in layer_indices),
    )


def omnivoice_q4_k_m_config(
    *,
    num_hidden_layers: int,
    audio_vocab_size: int,
    num_audio_codebook: int,
    audio_mask_id: int,
) -> Q4KMQuantizationConfig:
    return Q4KMQuantizationConfig(
        model_name=QUANTIZE_MODEL_NAME,
        gguf_arch=QUANTIZE_GGUF_ARCH,
        q6_tensor_names=omnivoice_q4_k_m_q6_tensor_names(num_hidden_layers),
        q8_tensor_names=Q8_TENSOR_NAMES,
        q8_tensor_prefixes=Q8_TENSOR_PREFIXES,
        safetensor_subdirs=("audio_tokenizer",),
        extra_uint32_metadata=(
            (f"{QUANTIZE_GGUF_ARCH}.audio_vocab_size", audio_vocab_size),
            (f"{QUANTIZE_GGUF_ARCH}.num_audio_codebook", num_audio_codebook),
            (f"{QUANTIZE_GGUF_ARCH}.audio_mask_id", audio_mask_id),
            (f"{QUANTIZE_GGUF_ARCH}.includes_audio_tokenizer", 1),
        ),
    )
