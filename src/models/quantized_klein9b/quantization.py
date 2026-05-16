from __future__ import annotations

from torch2vk.quantize import Q4KMQuantizationConfig, q4_k_m_more_bits_layer_indices


def klein9b_q4_k_m_config() -> Q4KMQuantizationConfig:
    q6_names = set[str]()
    for layer_idx in q4_k_m_more_bits_layer_indices(8):
        q6_names.update(
            {
                f"double_blocks.{layer_idx}.img_attn.qkv.weight",
                f"double_blocks.{layer_idx}.img_attn.proj.weight",
                f"double_blocks.{layer_idx}.txt_attn.qkv.weight",
                f"double_blocks.{layer_idx}.txt_attn.proj.weight",
            }
        )
    for layer_idx in q4_k_m_more_bits_layer_indices(24):
        q6_names.update(
            {
                f"single_blocks.{layer_idx}.linear1.weight",
                f"single_blocks.{layer_idx}.linear2.weight",
            }
        )
    return Q4KMQuantizationConfig(
        model_name="FLUX.2-klein-9B",
        gguf_arch="flux2-klein9b",
        q6_tensor_names=tuple(sorted(q6_names)),
        q8_tensor_names=(
            "img_in.weight",
            "txt_in.weight",
            "time_in.in_layer.weight",
            "time_in.out_layer.weight",
            "final_layer.linear.weight",
            "final_layer.adaLN_modulation.1.weight",
        ),
    )


def ae_q4_k_m_config() -> Q4KMQuantizationConfig:
    return Q4KMQuantizationConfig(
        model_name="FLUX.2 AutoEncoder",
        gguf_arch="flux2-ae",
        q8_tensor_prefixes=("",),
    )
