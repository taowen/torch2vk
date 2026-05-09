"""Qwen3-ASR runtime shader package."""

from __future__ import annotations

import sys

from torch2vk.runtime.shader import ShaderVariant, collect_shader_variants

from models.optimized_qwen3_asr.shaders.add_f32 import QWEN3_ASR_ADD_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.attention_f32 import QWEN3_ASR_ENCODER_ATTENTION_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.compact_after_cnn_f32 import QWEN3_ASR_COMPACT_AFTER_CNN_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.conv2d_gelu_f32 import QWEN3_ASR_CONV2D_GELU_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.conv_out_add_position_f32 import QWEN3_ASR_ADD_POSITION_F32, QWEN3_ASR_CONV_OUT_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.cu_seqlens_u32 import QWEN3_ASR_CU_SEQLENS_U32  # noqa: F401
from models.optimized_qwen3_asr.shaders.layer_norm_f32 import QWEN3_ASR_LAYER_NORM_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.linear_f32 import QWEN3_ASR_LINEAR_F32, QWEN3_ASR_LINEAR_GELU_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.pad_feature_f32 import QWEN3_ASR_PAD_FEATURE_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_add_3d_f32 import QWEN3_ASR_TEXT_ADD_3D_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_attention_decode_f32 import QWEN3_ASR_TEXT_ATTENTION_DECODE_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_attention_prefill_f32 import QWEN3_ASR_TEXT_ATTENTION_PREFILL_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_embed_lookup_f32 import QWEN3_ASR_TEXT_EMBED_LOOKUP_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_gate_up_swiglu_t1_f32 import QWEN3_ASR_TEXT_GATE_UP_SWIGLU_T1_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_kv_cache_write_f32 import QWEN3_ASR_TEXT_KV_CACHE_WRITE_DECODE_F32, QWEN3_ASR_TEXT_KV_CACHE_WRITE_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_linear_nobias_f32 import QWEN3_ASR_TEXT_LINEAR_NOBIAS_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_linear_nobias_t1_f32 import QWEN3_ASR_TEXT_LINEAR_NOBIAS_T1_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_linear_nobias_t1_splitk4_f32 import QWEN3_ASR_TEXT_LINEAR_NOBIAS_T1_SPLITK4_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_lm_head_select_t1_f32 import QWEN3_ASR_TEXT_LM_HEAD_SELECT_PARTIAL_T1_F32, QWEN3_ASR_TEXT_LM_HEAD_SELECT_REDUCE_T1_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_prefill_inputs_embeds_f32 import QWEN3_ASR_TEXT_PREFILL_INPUTS_EMBEDS_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_qk_norm_f32 import QWEN3_ASR_TEXT_QK_NORM_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_qkv_proj_t1_f32 import QWEN3_ASR_TEXT_QKV_PROJ_T1_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_rms_norm_f32 import QWEN3_ASR_TEXT_RMS_NORM_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_rope_f32 import QWEN3_ASR_TEXT_ROPE_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.text_swiglu_f32 import QWEN3_ASR_TEXT_SWIGLU_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.token_select_f32 import QWEN3_ASR_TOKEN_SELECT_GREEDY_F32  # noqa: F401
from models.optimized_qwen3_asr.shaders.token_store_f32 import QWEN3_ASR_TOKEN_STORE_EOS_F32, QWEN3_ASR_TOKEN_STORE_F32  # noqa: F401

_MODEL_SHADERS: dict[str, ShaderVariant] | None = None


def model_shaders() -> dict[str, ShaderVariant]:
    global _MODEL_SHADERS
    if _MODEL_SHADERS is None:
        _MODEL_SHADERS = collect_shader_variants(sys.modules[__name__])
    return _MODEL_SHADERS
