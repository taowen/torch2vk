"""Qwen3-ASR text prefill frame boundary."""

from __future__ import annotations

import numpy as np

from models.optimized_qwen3_asr.shaders.text_add_3d_f32 import QWEN3_ASR_TEXT_ADD_3D_F32
from models.optimized_qwen3_asr.shaders.text_attention_prefill_f32 import QWEN3_ASR_TEXT_ATTENTION_PREFILL_F32
from models.optimized_qwen3_asr.shaders.text_kv_cache_write_f32 import QWEN3_ASR_TEXT_KV_CACHE_WRITE_F32
from models.optimized_qwen3_asr.shaders.text_linear_nobias_f32 import QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32
from models.optimized_qwen3_asr.shaders.text_prefill_inputs_embeds_f32 import (
    QWEN3_ASR_TEXT_PREFILL_INPUTS_EMBEDS_F32,
)
from models.optimized_qwen3_asr.shaders.text_qk_norm_f32 import QWEN3_ASR_TEXT_QK_NORM_F32
from models.optimized_qwen3_asr.shaders.text_rms_norm_f32 import QWEN3_ASR_TEXT_RMS_NORM_F32
from models.optimized_qwen3_asr.shaders.text_rope_f32 import QWEN3_ASR_TEXT_ROPE_F32
from models.optimized_qwen3_asr.shaders.text_swiglu_f32 import QWEN3_ASR_TEXT_SWIGLU_F32
from models.optimized_qwen3_asr.tensors.text import Qwen3AsrTextPrefillTensors
from torch2vk.runtime.session import RuntimeSession


def run_qwen3_asr_text_prefill(
    rt: RuntimeSession,
    tensors: Qwen3AsrTextPrefillTensors,
    *,
    pytorch_compare: bool = True,
) -> None:
    """Frame boundary for prompt/audio merge and full-sequence text prefill."""
    initial_states = {}
    for layer in tensors.layers:
        for cache in (layer.key_cache, layer.value_cache):
            if cache.buffer is None:
                initial_states[cache] = np.zeros(cache.concrete_shape, dtype=np.float32)
    if initial_states:
        rt.initialize_request_state(initial_states)

    if pytorch_compare:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        frame_scope = rt.frame(
            "qwen3_asr.text_prefill",
            pytorch_model_class=Qwen3ASRForConditionalGeneration,
            pytorch_model_submodule="thinker",
            pytorch_input_prefixes=("qwen3_asr.text_prefill",),
            pytorch_cache_policy="hf_dynamic",
            pytorch_cache_namespace="qwen3_asr.text",
            pytorch_reset_cache=True,
        )
    else:
        frame_scope = rt.frame("qwen3_asr.text_prefill")

    with frame_scope:
        QWEN3_ASR_TEXT_PREFILL_INPUTS_EMBEDS_F32(
            rt,
            input_ids=tensors.input_ids,
            embed_tokens_weight=tensors.embed_tokens_weight,
            audio_features=tensors.audio_features,
            audio_scatter_mask=tensors.audio_scatter_mask,
            inputs_embeds=tensors.inputs_embeds,
        )

        hidden = tensors.inputs_embeds
        for layer in tensors.layers:
            QWEN3_ASR_TEXT_RMS_NORM_F32(
                rt, x=hidden, weight=layer.input_layernorm_weight, output=layer.input_layernorm,
            )
            QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.input_layernorm, weight=layer.q_proj_weight, output=layer.q_proj,
            )
            QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.input_layernorm, weight=layer.k_proj_weight, output=layer.k_proj,
            )
            QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.input_layernorm, weight=layer.v_proj_weight, output=layer.v_proj,
            )
            QWEN3_ASR_TEXT_QK_NORM_F32(
                rt, x=layer.q_proj, weight=layer.q_norm_weight, output=layer.q_normed,
            )
            QWEN3_ASR_TEXT_QK_NORM_F32(
                rt, x=layer.k_proj, weight=layer.k_norm_weight, output=layer.k_normed,
            )
            QWEN3_ASR_TEXT_ROPE_F32(
                rt, x=layer.q_normed, cos=tensors.rope_cos, sin=tensors.rope_sin,
                output=layer.q_roped,
            )
            QWEN3_ASR_TEXT_ROPE_F32(
                rt, x=layer.k_normed, cos=tensors.rope_cos, sin=tensors.rope_sin,
                output=layer.k_roped,
            )
            QWEN3_ASR_TEXT_KV_CACHE_WRITE_F32(
                rt, k=layer.k_roped, v=layer.v_proj,
                key_cache=layer.key_cache, value_cache=layer.value_cache,
            )
            QWEN3_ASR_TEXT_ATTENTION_PREFILL_F32(
                rt, q=layer.q_roped,
                key_cache=layer.key_cache, value_cache=layer.value_cache,
                output=layer.attention,
            )
            QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.attention, weight=layer.o_proj_weight, output=layer.o_proj,
            )
            QWEN3_ASR_TEXT_ADD_3D_F32(
                rt, x=hidden, y=layer.o_proj, output=layer.attn_residual,
            )
            QWEN3_ASR_TEXT_RMS_NORM_F32(
                rt, x=layer.attn_residual, weight=layer.post_attention_layernorm_weight,
                output=layer.post_attention_layernorm,
            )
            QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.post_attention_layernorm, weight=layer.gate_proj_weight,
                output=layer.gate_proj,
            )
            QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.post_attention_layernorm, weight=layer.up_proj_weight,
                output=layer.up_proj,
            )
            QWEN3_ASR_TEXT_SWIGLU_F32(
                rt, gate=layer.gate_proj, up=layer.up_proj, output=layer.swiglu,
            )
            QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(
                rt, x=layer.swiglu, weight=layer.down_proj_weight, output=layer.down_proj,
            )
            QWEN3_ASR_TEXT_ADD_3D_F32(
                rt, x=layer.attn_residual, y=layer.down_proj, output=layer.output,
            )
            hidden = layer.output

        QWEN3_ASR_TEXT_RMS_NORM_F32(
            rt, x=hidden, weight=tensors.norm_weight, output=tensors.final_norm,
        )
        QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32(
            rt, x=tensors.final_norm, weight=tensors.lm_head_weight, output=tensors.logits,
        )
