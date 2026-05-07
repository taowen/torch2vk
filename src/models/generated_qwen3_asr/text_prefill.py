"""Generated generated_qwen3_asr.text_prefill frame scaffold."""

from __future__ import annotations

from torch2vk.runtime.session import RuntimeSession
from torch2vk.exportv2.shaders import (
    TEXT_ADD_3D_F32,
    TEXT_ATTENTION_PREFILL_F32,
    TEXT_KV_CACHE_WRITE_F32,
    TEXT_LINEAR_NOBIAS_F32,
    TEXT_PREFILL_INPUTS_EMBEDS_F32,
    TEXT_QK_NORM_F32,
    TEXT_RMS_NORM_F32,
    TEXT_ROPE_F32,
    TEXT_SWIGLU_F32,
)
from models.generated_qwen3_asr.tensors.text import GeneratedQwen3AsrTextPrefillTensors


def run_generated_qwen3_asr_text_prefill(
    rt: RuntimeSession,
    tensors: GeneratedQwen3AsrTextPrefillTensors,
    **_kw,
) -> None:
    with rt.frame("generated_qwen3_asr.text_prefill"):
        TEXT_PREFILL_INPUTS_EMBEDS_F32(
            rt,
            input_ids=tensors.input_ids,
            embed_tokens_weight=tensors.embed_tokens_weight,
            audio_features=tensors.audio_features,
            audio_scatter_mask=tensors.audio_scatter_mask,
            inputs_embeds=tensors.inputs_embeds,
        )
        hidden = tensors.inputs_embeds
        for layer in tensors.layers:
            TEXT_RMS_NORM_F32(
                rt, x=hidden, weight=layer.input_layernorm_weight, output=layer.input_layernorm
            )
            TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.input_layernorm, weight=layer.q_proj_weight, output=layer.q_proj
            )
            TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.input_layernorm, weight=layer.k_proj_weight, output=layer.k_proj
            )
            TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.input_layernorm, weight=layer.v_proj_weight, output=layer.v_proj
            )
            TEXT_QK_NORM_F32(rt, x=layer.q_proj, weight=layer.q_norm_weight, output=layer.q_normed)
            TEXT_QK_NORM_F32(rt, x=layer.k_proj, weight=layer.k_norm_weight, output=layer.k_normed)
            TEXT_ROPE_F32(
                rt,
                x=layer.q_normed,
                cos=tensors.rope_cos,
                sin=tensors.rope_sin,
                output=layer.q_roped,
            )
            TEXT_ROPE_F32(
                rt,
                x=layer.k_normed,
                cos=tensors.rope_cos,
                sin=tensors.rope_sin,
                output=layer.k_roped,
            )
            TEXT_KV_CACHE_WRITE_F32(
                rt,
                k=layer.k_roped,
                v=layer.v_proj,
                key_cache=layer.key_cache,
                value_cache=layer.value_cache,
            )
            TEXT_ATTENTION_PREFILL_F32(
                rt,
                q=layer.q_roped,
                key_cache=layer.key_cache,
                value_cache=layer.value_cache,
                output=layer.attention,
            )
            TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.attention, weight=layer.o_proj_weight, output=layer.o_proj
            )
            TEXT_ADD_3D_F32(rt, x=hidden, y=layer.o_proj, output=layer.attn_residual)
            TEXT_RMS_NORM_F32(
                rt,
                x=layer.attn_residual,
                weight=layer.post_attention_layernorm_weight,
                output=layer.post_attention_layernorm,
            )
            TEXT_LINEAR_NOBIAS_F32(
                rt,
                x=layer.post_attention_layernorm,
                weight=layer.gate_proj_weight,
                output=layer.gate_proj,
            )
            TEXT_LINEAR_NOBIAS_F32(
                rt,
                x=layer.post_attention_layernorm,
                weight=layer.up_proj_weight,
                output=layer.up_proj,
            )
            TEXT_SWIGLU_F32(rt, gate=layer.gate_proj, up=layer.up_proj, output=layer.swiglu)
            TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.swiglu, weight=layer.down_proj_weight, output=layer.down_proj
            )
            TEXT_ADD_3D_F32(rt, x=layer.attn_residual, y=layer.down_proj, output=layer.output)
            hidden = layer.output
        TEXT_RMS_NORM_F32(rt, x=hidden, weight=tensors.norm_weight, output=tensors.final_norm)
        TEXT_LINEAR_NOBIAS_F32(
            rt, x=tensors.final_norm, weight=tensors.lm_head_weight, output=tensors.logits
        )
