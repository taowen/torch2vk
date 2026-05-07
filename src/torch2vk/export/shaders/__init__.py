"""Model-agnostic shaders owned by torch2vk.export."""

from torch2vk.export.shaders.aten_embedding_3d_f32 import ATEN_EMBEDDING_3D_F32
from torch2vk.export.shaders.aten_embedding_f32 import ATEN_EMBEDDING_F32
from torch2vk.export.shaders.aten_select_int_i64 import ATEN_SELECT_INT_I64
from torch2vk.export.shaders.aten_shifted_ids_i64 import ATEN_SHIFTED_IDS_I64
from torch2vk.export.shaders.aten_sum_dim1_f32 import ATEN_SUM_DIM1_F32
from torch2vk.export.shaders.aten_where_f32 import ATEN_WHERE_F32
from torch2vk.export.shaders.add_f32 import ADD_F32
from torch2vk.export.shaders.attention_f32 import ENCODER_ATTENTION_F32
from torch2vk.export.shaders.compact_after_cnn_f32 import COMPACT_AFTER_CNN_F32
from torch2vk.export.shaders.conv2d_gelu_f32 import CONV2D_GELU_F32
from torch2vk.export.shaders.conv_out_add_position_f32 import (
    ADD_POSITION_F32,
    CONV_OUT_F32,
)
from torch2vk.export.shaders.cu_seqlens_u32 import CU_SEQLENS_U32
from torch2vk.export.shaders.layer_norm_f32 import LAYER_NORM_F32
from torch2vk.export.shaders.linear_f32 import LINEAR_F32, LINEAR_GELU_F32
from torch2vk.export.shaders.pad_feature_f32 import PAD_FEATURE_F32
from torch2vk.export.shaders.text_embed_lookup_f32 import TEXT_EMBED_LOOKUP_F32
from torch2vk.export.shaders.text_add_3d_f32 import TEXT_ADD_3D_F32
from torch2vk.export.shaders.text_attention_decode_f32 import TEXT_ATTENTION_DECODE_F32
from torch2vk.export.shaders.text_attention_prefill_f32 import TEXT_ATTENTION_PREFILL_F32
from torch2vk.export.shaders.text_kv_cache_write_f32 import (
    TEXT_KV_CACHE_WRITE_DECODE_F32,
    TEXT_KV_CACHE_WRITE_F32,
)
from torch2vk.export.shaders.text_linear_nobias_f32 import TEXT_LINEAR_NOBIAS_F32
from torch2vk.export.shaders.text_linear_nobias_t1_f32 import TEXT_LINEAR_NOBIAS_T1_F32
from torch2vk.export.shaders.text_linear_nobias_t1_splitk4_f32 import TEXT_LINEAR_NOBIAS_T1_SPLITK4_F32
from torch2vk.export.shaders.text_lm_head_select_t1_f32 import (
    TEXT_LM_HEAD_SELECT_PARTIAL_T1_F32,
    TEXT_LM_HEAD_SELECT_REDUCE_T1_F32,
)
from torch2vk.export.shaders.text_prefill_inputs_embeds_f32 import TEXT_PREFILL_INPUTS_EMBEDS_F32
from torch2vk.export.shaders.text_qkv_proj_t1_f32 import TEXT_QKV_PROJ_T1_F32
from torch2vk.export.shaders.text_qk_norm_f32 import TEXT_QK_NORM_F32
from torch2vk.export.shaders.text_rms_norm_f32 import TEXT_RMS_NORM_F32
from torch2vk.export.shaders.text_rope_f32 import TEXT_ROPE_F32
from torch2vk.export.shaders.text_swiglu_f32 import TEXT_SWIGLU_F32
from torch2vk.export.shaders.text_gate_up_swiglu_t1_f32 import TEXT_GATE_UP_SWIGLU_T1_F32
from torch2vk.export.shaders.token_select_greedy_f32 import TOKEN_SELECT_GREEDY_F32
from torch2vk.export.shaders.rope_table_f32 import ROPE_TABLE_F32
from torch2vk.export.shaders.token_store_f32 import TOKEN_STORE_EOS_F32, TOKEN_STORE_F32

__all__ = [
    "ATEN_SELECT_INT_I64",
    "ATEN_EMBEDDING_F32",
    "ATEN_SHIFTED_IDS_I64",
    "ATEN_EMBEDDING_3D_F32",
    "ATEN_SUM_DIM1_F32",
    "ATEN_WHERE_F32",
    "PAD_FEATURE_F32",
    "CONV2D_GELU_F32",
    "CONV_OUT_F32",
    "ADD_POSITION_F32",
    "COMPACT_AFTER_CNN_F32",
    "CU_SEQLENS_U32",
    "LAYER_NORM_F32",
    "LINEAR_F32",
    "LINEAR_GELU_F32",
    "ENCODER_ATTENTION_F32",
    "ADD_F32",
    "TEXT_PREFILL_INPUTS_EMBEDS_F32",
    "TEXT_ADD_3D_F32",
    "TEXT_QK_NORM_F32",
    "TEXT_ROPE_F32",
    "TEXT_SWIGLU_F32",
    "TEXT_ATTENTION_PREFILL_F32",
    "TEXT_ATTENTION_DECODE_F32",
    "TEXT_KV_CACHE_WRITE_F32",
    "TEXT_KV_CACHE_WRITE_DECODE_F32",
    "TEXT_RMS_NORM_F32",
    "TEXT_LINEAR_NOBIAS_F32",
    "TEXT_LINEAR_NOBIAS_T1_F32",
    "TEXT_LINEAR_NOBIAS_T1_SPLITK4_F32",
    "TEXT_LM_HEAD_SELECT_PARTIAL_T1_F32",
    "TEXT_LM_HEAD_SELECT_REDUCE_T1_F32",
    "TEXT_QKV_PROJ_T1_F32",
    "TEXT_GATE_UP_SWIGLU_T1_F32",
    "TEXT_EMBED_LOOKUP_F32",
    "TOKEN_SELECT_GREEDY_F32",
    "ROPE_TABLE_F32",
    "TOKEN_STORE_F32",
    "TOKEN_STORE_EOS_F32",
]

from torch2vk.runtime.shader import ShaderVariant as _ShaderVariant

_EXPORT_SHADER_VARIANTS: dict[str, _ShaderVariant] = {
    name: value
    for name in __all__
    if isinstance(value := globals()[name], _ShaderVariant)
}

