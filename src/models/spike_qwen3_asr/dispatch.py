"""Generated dispatch functions for all submodules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from models.qwen3_asr.shaders.token_select_f32 import QWEN3_ASR_TOKEN_SELECT_GREEDY_F32
from models.qwen3_asr.shaders.token_store_f32 import QWEN3_ASR_TOKEN_STORE_EOS_F32
from models.spike_qwen3_asr.shaders.conv2d2_export_conv2d_f32 import CONV2D2_EXPORT_CONV2D_F32
from models.spike_qwen3_asr.shaders.conv2d3_export_conv2d_f32 import CONV2D3_EXPORT_CONV2D_F32
from models.spike_qwen3_asr.shaders.decode_embed_export_embedding_f32 import DECODE_EMBED_EXPORT_EMBEDDING_F32
from models.spike_qwen3_asr.shaders.decode_layer_export_add_f32 import DECODE_LAYER_EXPORT_ADD_F32
from models.spike_qwen3_asr.shaders.decode_layer_export_cat_f32 import DECODE_LAYER_EXPORT_CAT_F32
from models.spike_qwen3_asr.shaders.decode_layer_export_cat_f32_32 import DECODE_LAYER_EXPORT_CAT_F32_32
from models.spike_qwen3_asr.shaders.decode_layer_export_linear_nobias_f32 import DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32
from models.spike_qwen3_asr.shaders.decode_layer_export_linear_nobias_f32_14 import DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_14
from models.spike_qwen3_asr.shaders.decode_layer_export_linear_nobias_f32_22 import DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_22
from models.spike_qwen3_asr.shaders.decode_layer_export_linear_nobias_f32_37 import DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_37
from models.spike_qwen3_asr.shaders.decode_layer_export_linear_nobias_f32_39 import DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_39
from models.spike_qwen3_asr.shaders.decode_layer_export_linear_nobias_f32_41 import DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_41
from models.spike_qwen3_asr.shaders.decode_layer_export_linear_nobias_f32_43 import DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_43
from models.spike_qwen3_asr.shaders.decode_layer_export_mean_dim_f32 import DECODE_LAYER_EXPORT_MEAN_DIM_F32
from models.spike_qwen3_asr.shaders.decode_layer_export_mean_dim_f32_16 import DECODE_LAYER_EXPORT_MEAN_DIM_F32_16
from models.spike_qwen3_asr.shaders.decode_layer_export_mean_dim_f32_8 import DECODE_LAYER_EXPORT_MEAN_DIM_F32_8
from models.spike_qwen3_asr.shaders.decode_layer_export_mul_broadcast_inner import DECODE_LAYER_EXPORT_MUL_BROADCAST_INNER
from models.spike_qwen3_asr.shaders.decode_layer_export_mul_broadcast_inner_29 import DECODE_LAYER_EXPORT_MUL_BROADCAST_INNER_29
from models.spike_qwen3_asr.shaders.decode_layer_export_mul_broadcast_inner_33 import DECODE_LAYER_EXPORT_MUL_BROADCAST_INNER_33
from models.spike_qwen3_asr.shaders.decode_layer_export_mul_broadcast_last import DECODE_LAYER_EXPORT_MUL_BROADCAST_LAST
from models.spike_qwen3_asr.shaders.decode_layer_export_mul_broadcast_last_11 import DECODE_LAYER_EXPORT_MUL_BROADCAST_LAST_11
from models.spike_qwen3_asr.shaders.decode_layer_export_mul_broadcast_last_19 import DECODE_LAYER_EXPORT_MUL_BROADCAST_LAST_19
from models.spike_qwen3_asr.shaders.decode_layer_export_slice_f32 import DECODE_LAYER_EXPORT_SLICE_F32
from models.spike_qwen3_asr.shaders.decode_layer_export_slice_f32_25 import DECODE_LAYER_EXPORT_SLICE_F32_25
from models.spike_qwen3_asr.shaders.decode_layer_export_slice_f32_30 import DECODE_LAYER_EXPORT_SLICE_F32_30
from models.spike_qwen3_asr.shaders.decode_layer_export_slice_f32_31 import DECODE_LAYER_EXPORT_SLICE_F32_31
from models.spike_qwen3_asr.shaders.decode_lm_head_export_linear_nobias_f32 import DECODE_LM_HEAD_EXPORT_LINEAR_NOBIAS_F32
from models.spike_qwen3_asr.shaders.decode_norm_export_mean_dim_f32 import DECODE_NORM_EXPORT_MEAN_DIM_F32
from models.spike_qwen3_asr.shaders.decode_norm_export_mul_broadcast_last import DECODE_NORM_EXPORT_MUL_BROADCAST_LAST
from models.spike_qwen3_asr.shaders.encoder_layer_export_add_f32 import ENCODER_LAYER_EXPORT_ADD_F32
from models.spike_qwen3_asr.shaders.encoder_layer_export_gelu_f32 import ENCODER_LAYER_EXPORT_GELU_F32
from models.spike_qwen3_asr.shaders.export_add_f32 import EXPORT_ADD_F32
from models.spike_qwen3_asr.shaders.export_add_f32_38 import EXPORT_ADD_F32_38
from models.spike_qwen3_asr.shaders.export_add_f32_44 import EXPORT_ADD_F32_44
from models.spike_qwen3_asr.shaders.export_add_scalar import EXPORT_ADD_SCALAR
from models.spike_qwen3_asr.shaders.export_add_scalar_17 import EXPORT_ADD_SCALAR_17
from models.spike_qwen3_asr.shaders.export_add_scalar_9 import EXPORT_ADD_SCALAR_9
from models.spike_qwen3_asr.shaders.export_cat_f32 import EXPORT_CAT_F32
from models.spike_qwen3_asr.shaders.export_cat_f32_32 import EXPORT_CAT_F32_32
from models.spike_qwen3_asr.shaders.export_conv2d_f32 import EXPORT_CONV2D_F32
from models.spike_qwen3_asr.shaders.export_embedding_f32 import EXPORT_EMBEDDING_F32
from models.spike_qwen3_asr.shaders.export_gelu_f32 import EXPORT_GELU_F32
from models.spike_qwen3_asr.shaders.export_index_copy_f32_7ba4f1ff13 import EXPORT_INDEX_COPY_F32_7BA4F1FF13
from models.spike_qwen3_asr.shaders.export_index_select_f32_c6680f8d95 import EXPORT_INDEX_SELECT_F32_C6680F8D95
from models.spike_qwen3_asr.shaders.export_kv_cache_write_f32 import EXPORT_KV_CACHE_WRITE_F32
from models.spike_qwen3_asr.shaders.export_layer_norm_f32 import EXPORT_LAYER_NORM_F32
from models.spike_qwen3_asr.shaders.export_linear_bias_f32 import EXPORT_LINEAR_BIAS_F32
from models.spike_qwen3_asr.shaders.export_linear_bias_f32_10 import EXPORT_LINEAR_BIAS_F32_10
from models.spike_qwen3_asr.shaders.export_linear_bias_f32_12 import EXPORT_LINEAR_BIAS_F32_12
from models.spike_qwen3_asr.shaders.export_linear_nobias_f32 import EXPORT_LINEAR_NOBIAS_F32
from models.spike_qwen3_asr.shaders.export_linear_nobias_f32_14 import EXPORT_LINEAR_NOBIAS_F32_14
from models.spike_qwen3_asr.shaders.export_linear_nobias_f32_22 import EXPORT_LINEAR_NOBIAS_F32_22
from models.spike_qwen3_asr.shaders.export_linear_nobias_f32_37 import EXPORT_LINEAR_NOBIAS_F32_37
from models.spike_qwen3_asr.shaders.export_linear_nobias_f32_39 import EXPORT_LINEAR_NOBIAS_F32_39
from models.spike_qwen3_asr.shaders.export_linear_nobias_f32_41 import EXPORT_LINEAR_NOBIAS_F32_41
from models.spike_qwen3_asr.shaders.export_linear_nobias_f32_43 import EXPORT_LINEAR_NOBIAS_F32_43
from models.spike_qwen3_asr.shaders.export_mean_dim_f32 import EXPORT_MEAN_DIM_F32
from models.spike_qwen3_asr.shaders.export_mean_dim_f32_16 import EXPORT_MEAN_DIM_F32_16
from models.spike_qwen3_asr.shaders.export_mean_dim_f32_8 import EXPORT_MEAN_DIM_F32_8
from models.spike_qwen3_asr.shaders.export_mul_broadcast_inner import EXPORT_MUL_BROADCAST_INNER
from models.spike_qwen3_asr.shaders.export_mul_broadcast_inner_29 import EXPORT_MUL_BROADCAST_INNER_29
from models.spike_qwen3_asr.shaders.export_mul_broadcast_inner_33 import EXPORT_MUL_BROADCAST_INNER_33
from models.spike_qwen3_asr.shaders.export_mul_broadcast_last import EXPORT_MUL_BROADCAST_LAST
from models.spike_qwen3_asr.shaders.export_mul_broadcast_last_11 import EXPORT_MUL_BROADCAST_LAST_11
from models.spike_qwen3_asr.shaders.export_mul_broadcast_last_19 import EXPORT_MUL_BROADCAST_LAST_19
from models.spike_qwen3_asr.shaders.export_mul_f32 import EXPORT_MUL_F32
from models.spike_qwen3_asr.shaders.export_mul_left_broadcast import EXPORT_MUL_LEFT_BROADCAST
from models.spike_qwen3_asr.shaders.export_mul_left_broadcast_12 import EXPORT_MUL_LEFT_BROADCAST_12
from models.spike_qwen3_asr.shaders.export_mul_left_broadcast_20 import EXPORT_MUL_LEFT_BROADCAST_20
from models.spike_qwen3_asr.shaders.export_neg_f32 import EXPORT_NEG_F32
from models.spike_qwen3_asr.shaders.export_pow_scalar_f32 import EXPORT_POW_SCALAR_F32
from models.spike_qwen3_asr.shaders.export_pow_scalar_f32_15 import EXPORT_POW_SCALAR_F32_15
from models.spike_qwen3_asr.shaders.export_pow_scalar_f32_7 import EXPORT_POW_SCALAR_F32_7
from models.spike_qwen3_asr.shaders.export_rsqrt_f32 import EXPORT_RSQRT_F32
from models.spike_qwen3_asr.shaders.export_rsqrt_f32_10 import EXPORT_RSQRT_F32_10
from models.spike_qwen3_asr.shaders.export_rsqrt_f32_18 import EXPORT_RSQRT_F32_18
from models.spike_qwen3_asr.shaders.export_sdpa_causal_f32 import EXPORT_SDPA_CAUSAL_F32
from models.spike_qwen3_asr.shaders.export_sdpa_decode_cache_f32 import EXPORT_SDPA_DECODE_CACHE_F32
from models.spike_qwen3_asr.shaders.export_sdpa_masked_f32 import EXPORT_SDPA_MASKED_F32
from models.spike_qwen3_asr.shaders.export_silu_f32 import EXPORT_SILU_F32
from models.spike_qwen3_asr.shaders.export_slice_f32 import EXPORT_SLICE_F32
from models.spike_qwen3_asr.shaders.export_slice_f32_25 import EXPORT_SLICE_F32_25
from models.spike_qwen3_asr.shaders.export_slice_f32_30 import EXPORT_SLICE_F32_30
from models.spike_qwen3_asr.shaders.export_slice_f32_31 import EXPORT_SLICE_F32_31
from models.spike_qwen3_asr.shaders.export_transpose_f32_48d16b9b88 import EXPORT_TRANSPOSE_F32_48D16B9B88
from models.spike_qwen3_asr.shaders.export_transpose_f32_6392135058 import EXPORT_TRANSPOSE_F32_6392135058
from models.spike_qwen3_asr.shaders.export_transpose_f32_6a3397f037 import EXPORT_TRANSPOSE_F32_6A3397F037
from models.spike_qwen3_asr.shaders.export_transpose_f32_8e058a050e import EXPORT_TRANSPOSE_F32_8E058A050E
from models.spike_qwen3_asr.shaders.export_transpose_f32_9884b7b82d import EXPORT_TRANSPOSE_F32_9884B7B82D
from models.spike_qwen3_asr.shaders.export_transpose_f32_9e00ca2f33 import EXPORT_TRANSPOSE_F32_9E00CA2F33
from models.spike_qwen3_asr.shaders.export_transpose_f32_9e77b1cee2 import EXPORT_TRANSPOSE_F32_9E77B1CEE2
from models.spike_qwen3_asr.shaders.export_transpose_f32_d509518b4f import EXPORT_TRANSPOSE_F32_D509518B4F
from models.spike_qwen3_asr.shaders.export_transpose_f32_d95ce920ac import EXPORT_TRANSPOSE_F32_D95CE920AC
from models.spike_qwen3_asr.shaders.lm_head_export_linear_nobias_f32 import LM_HEAD_EXPORT_LINEAR_NOBIAS_F32
from models.spike_qwen3_asr.shaders.proj1_export_gelu_f32 import PROJ1_EXPORT_GELU_F32
from models.spike_qwen3_asr.shaders.proj2_export_linear_bias_f32 import PROJ2_EXPORT_LINEAR_BIAS_F32
from models.spike_qwen3_asr.shaders.text_layer_export_add_f32 import TEXT_LAYER_EXPORT_ADD_F32
from models.spike_qwen3_asr.shaders.text_layer_export_linear_nobias_f32 import TEXT_LAYER_EXPORT_LINEAR_NOBIAS_F32
from models.spike_qwen3_asr.tensors.audio_tower import AudioPositionCompactTensors, Conv2d1Tensors, Conv2d2Tensors, Conv2d3Tensors, ConvOutTensors, LnPostTensors, Proj1Tensors, Proj2Tensors
from models.spike_qwen3_asr.tensors.decode import DecodeEmbedTensors, DecodeLmHeadTensors, DecodeNormTensors
from models.spike_qwen3_asr.tensors.decode_layer import DecodeLayerTensors
from models.spike_qwen3_asr.tensors.encoder_layer import EncoderLayerTensors
from models.spike_qwen3_asr.tensors.text import AudioInjectTensors, EmbedTokensTensors, LmHeadTensors, TextNormTensors
from models.spike_qwen3_asr.tensors.text_layer import TextLayerTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.shader import ShaderVariant
from torch2vk.runtime.session import RuntimeSession


SHADER_VARIANTS_BY_NAME: dict[str, ShaderVariant] = {
    "conv2d2_export_conv2d_f32": CONV2D2_EXPORT_CONV2D_F32,
    "conv2d3_export_conv2d_f32": CONV2D3_EXPORT_CONV2D_F32,
    "decode_embed_export_embedding_f32": DECODE_EMBED_EXPORT_EMBEDDING_F32,
    "decode_layer_export_add_f32": DECODE_LAYER_EXPORT_ADD_F32,
    "decode_layer_export_cat_f32": DECODE_LAYER_EXPORT_CAT_F32,
    "decode_layer_export_cat_f32_32": DECODE_LAYER_EXPORT_CAT_F32_32,
    "decode_layer_export_linear_nobias_f32": DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32,
    "decode_layer_export_linear_nobias_f32_14": DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_14,
    "decode_layer_export_linear_nobias_f32_22": DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_22,
    "decode_layer_export_linear_nobias_f32_37": DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_37,
    "decode_layer_export_linear_nobias_f32_39": DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_39,
    "decode_layer_export_linear_nobias_f32_41": DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_41,
    "decode_layer_export_linear_nobias_f32_43": DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_43,
    "decode_layer_export_mean_dim_f32": DECODE_LAYER_EXPORT_MEAN_DIM_F32,
    "decode_layer_export_mean_dim_f32_16": DECODE_LAYER_EXPORT_MEAN_DIM_F32_16,
    "decode_layer_export_mean_dim_f32_8": DECODE_LAYER_EXPORT_MEAN_DIM_F32_8,
    "decode_layer_export_mul_broadcast_inner": DECODE_LAYER_EXPORT_MUL_BROADCAST_INNER,
    "decode_layer_export_mul_broadcast_inner_29": DECODE_LAYER_EXPORT_MUL_BROADCAST_INNER_29,
    "decode_layer_export_mul_broadcast_inner_33": DECODE_LAYER_EXPORT_MUL_BROADCAST_INNER_33,
    "decode_layer_export_mul_broadcast_last": DECODE_LAYER_EXPORT_MUL_BROADCAST_LAST,
    "decode_layer_export_mul_broadcast_last_11": DECODE_LAYER_EXPORT_MUL_BROADCAST_LAST_11,
    "decode_layer_export_mul_broadcast_last_19": DECODE_LAYER_EXPORT_MUL_BROADCAST_LAST_19,
    "decode_layer_export_slice_f32": DECODE_LAYER_EXPORT_SLICE_F32,
    "decode_layer_export_slice_f32_25": DECODE_LAYER_EXPORT_SLICE_F32_25,
    "decode_layer_export_slice_f32_30": DECODE_LAYER_EXPORT_SLICE_F32_30,
    "decode_layer_export_slice_f32_31": DECODE_LAYER_EXPORT_SLICE_F32_31,
    "decode_lm_head_export_linear_nobias_f32": DECODE_LM_HEAD_EXPORT_LINEAR_NOBIAS_F32,
    "decode_norm_export_mean_dim_f32": DECODE_NORM_EXPORT_MEAN_DIM_F32,
    "decode_norm_export_mul_broadcast_last": DECODE_NORM_EXPORT_MUL_BROADCAST_LAST,
    "encoder_layer_export_add_f32": ENCODER_LAYER_EXPORT_ADD_F32,
    "encoder_layer_export_gelu_f32": ENCODER_LAYER_EXPORT_GELU_F32,
    "export_add_f32": EXPORT_ADD_F32,
    "export_add_f32_38": EXPORT_ADD_F32_38,
    "export_add_f32_44": EXPORT_ADD_F32_44,
    "export_add_scalar": EXPORT_ADD_SCALAR,
    "export_add_scalar_17": EXPORT_ADD_SCALAR_17,
    "export_add_scalar_9": EXPORT_ADD_SCALAR_9,
    "export_cat_f32": EXPORT_CAT_F32,
    "export_cat_f32_32": EXPORT_CAT_F32_32,
    "export_conv2d_f32": EXPORT_CONV2D_F32,
    "export_embedding_f32": EXPORT_EMBEDDING_F32,
    "export_gelu_f32": EXPORT_GELU_F32,
    "export_index_copy_f32_7ba4f1ff13": EXPORT_INDEX_COPY_F32_7BA4F1FF13,
    "export_index_select_f32_c6680f8d95": EXPORT_INDEX_SELECT_F32_C6680F8D95,
    "export_kv_cache_write_f32": EXPORT_KV_CACHE_WRITE_F32,
    "export_layer_norm_f32": EXPORT_LAYER_NORM_F32,
    "export_linear_bias_f32": EXPORT_LINEAR_BIAS_F32,
    "export_linear_bias_f32_10": EXPORT_LINEAR_BIAS_F32_10,
    "export_linear_bias_f32_12": EXPORT_LINEAR_BIAS_F32_12,
    "export_linear_nobias_f32": EXPORT_LINEAR_NOBIAS_F32,
    "export_linear_nobias_f32_14": EXPORT_LINEAR_NOBIAS_F32_14,
    "export_linear_nobias_f32_22": EXPORT_LINEAR_NOBIAS_F32_22,
    "export_linear_nobias_f32_37": EXPORT_LINEAR_NOBIAS_F32_37,
    "export_linear_nobias_f32_39": EXPORT_LINEAR_NOBIAS_F32_39,
    "export_linear_nobias_f32_41": EXPORT_LINEAR_NOBIAS_F32_41,
    "export_linear_nobias_f32_43": EXPORT_LINEAR_NOBIAS_F32_43,
    "export_mean_dim_f32": EXPORT_MEAN_DIM_F32,
    "export_mean_dim_f32_16": EXPORT_MEAN_DIM_F32_16,
    "export_mean_dim_f32_8": EXPORT_MEAN_DIM_F32_8,
    "export_mul_broadcast_inner": EXPORT_MUL_BROADCAST_INNER,
    "export_mul_broadcast_inner_29": EXPORT_MUL_BROADCAST_INNER_29,
    "export_mul_broadcast_inner_33": EXPORT_MUL_BROADCAST_INNER_33,
    "export_mul_broadcast_last": EXPORT_MUL_BROADCAST_LAST,
    "export_mul_broadcast_last_11": EXPORT_MUL_BROADCAST_LAST_11,
    "export_mul_broadcast_last_19": EXPORT_MUL_BROADCAST_LAST_19,
    "export_mul_f32": EXPORT_MUL_F32,
    "export_mul_left_broadcast": EXPORT_MUL_LEFT_BROADCAST,
    "export_mul_left_broadcast_12": EXPORT_MUL_LEFT_BROADCAST_12,
    "export_mul_left_broadcast_20": EXPORT_MUL_LEFT_BROADCAST_20,
    "export_neg_f32": EXPORT_NEG_F32,
    "export_pow_scalar_f32": EXPORT_POW_SCALAR_F32,
    "export_pow_scalar_f32_15": EXPORT_POW_SCALAR_F32_15,
    "export_pow_scalar_f32_7": EXPORT_POW_SCALAR_F32_7,
    "export_rsqrt_f32": EXPORT_RSQRT_F32,
    "export_rsqrt_f32_10": EXPORT_RSQRT_F32_10,
    "export_rsqrt_f32_18": EXPORT_RSQRT_F32_18,
    "export_sdpa_causal_f32": EXPORT_SDPA_CAUSAL_F32,
    "export_sdpa_decode_cache_f32": EXPORT_SDPA_DECODE_CACHE_F32,
    "export_sdpa_masked_f32": EXPORT_SDPA_MASKED_F32,
    "export_silu_f32": EXPORT_SILU_F32,
    "export_slice_f32": EXPORT_SLICE_F32,
    "export_slice_f32_25": EXPORT_SLICE_F32_25,
    "export_slice_f32_30": EXPORT_SLICE_F32_30,
    "export_slice_f32_31": EXPORT_SLICE_F32_31,
    "export_transpose_f32_48d16b9b88": EXPORT_TRANSPOSE_F32_48D16B9B88,
    "export_transpose_f32_6392135058": EXPORT_TRANSPOSE_F32_6392135058,
    "export_transpose_f32_6a3397f037": EXPORT_TRANSPOSE_F32_6A3397F037,
    "export_transpose_f32_8e058a050e": EXPORT_TRANSPOSE_F32_8E058A050E,
    "export_transpose_f32_9884b7b82d": EXPORT_TRANSPOSE_F32_9884B7B82D,
    "export_transpose_f32_9e00ca2f33": EXPORT_TRANSPOSE_F32_9E00CA2F33,
    "export_transpose_f32_9e77b1cee2": EXPORT_TRANSPOSE_F32_9E77B1CEE2,
    "export_transpose_f32_d509518b4f": EXPORT_TRANSPOSE_F32_D509518B4F,
    "export_transpose_f32_d95ce920ac": EXPORT_TRANSPOSE_F32_D95CE920AC,
    "lm_head_export_linear_nobias_f32": LM_HEAD_EXPORT_LINEAR_NOBIAS_F32,
    "proj1_export_gelu_f32": PROJ1_EXPORT_GELU_F32,
    "proj2_export_linear_bias_f32": PROJ2_EXPORT_LINEAR_BIAS_F32,
    "text_layer_export_add_f32": TEXT_LAYER_EXPORT_ADD_F32,
    "text_layer_export_linear_nobias_f32": TEXT_LAYER_EXPORT_LINEAR_NOBIAS_F32,
    QWEN3_ASR_TOKEN_SELECT_GREEDY_F32.name: QWEN3_ASR_TOKEN_SELECT_GREEDY_F32,
    QWEN3_ASR_TOKEN_STORE_EOS_F32.name: QWEN3_ASR_TOKEN_STORE_EOS_F32,
}


def run_conv2d1(rt: RuntimeSession, tensors: Conv2d1Tensors) -> None:
    EXPORT_CONV2D_F32(rt, x=tensors.x, weight=tensors.p_weight, bias=tensors.p_bias, output=tensors.conv2d)
    EXPORT_GELU_F32(rt, x=tensors.conv2d, output=tensors.gelu)


def run_conv2d2(rt: RuntimeSession, tensors: Conv2d2Tensors) -> None:
    CONV2D2_EXPORT_CONV2D_F32(rt, x=tensors.x, weight=tensors.p_weight, bias=tensors.p_bias, output=tensors.conv2d)
    EXPORT_GELU_F32(rt, x=tensors.conv2d, output=tensors.gelu)


def run_conv2d3(rt: RuntimeSession, tensors: Conv2d3Tensors) -> None:
    CONV2D3_EXPORT_CONV2D_F32(rt, x=tensors.x, weight=tensors.p_weight, bias=tensors.p_bias, output=tensors.conv2d)
    EXPORT_GELU_F32(rt, x=tensors.conv2d, output=tensors.gelu)


def run_conv_out(rt: RuntimeSession, tensors: ConvOutTensors) -> None:
    _alias(rt, tensors.x, tensors.reshape)
    EXPORT_TRANSPOSE_F32_D95CE920AC(rt, x=tensors.reshape, output=tensors.transpose)
    EXPORT_LINEAR_NOBIAS_F32(rt, x=tensors.transpose, weight=tensors.p_weight, output=tensors.linear)


def run_audio_position_compact(rt: RuntimeSession, tensors: AudioPositionCompactTensors) -> None:
    EXPORT_ADD_F32(rt, x=tensors.x, y=tensors.position_embedding, output=tensors.add)
    _alias(rt, tensors.add, tensors.reshape)
    EXPORT_INDEX_SELECT_F32_C6680F8D95(rt, x=tensors.reshape, index=tensors.compact_index, output=tensors.index_select)


def run_encoder_layer(rt: RuntimeSession, tensors: EncoderLayerTensors) -> None:
    EXPORT_LAYER_NORM_F32(rt, x=tensors.hidden_states, weight=tensors.p_attn_layer_norm_weight, bias=tensors.p_attn_layer_norm_bias, output=tensors.layer_norm)
    EXPORT_LINEAR_BIAS_F32(rt, x=tensors.layer_norm, weight=tensors.p_attn_q_proj_weight, bias=tensors.p_attn_q_proj_bias, output=tensors.linear)
    _alias(rt, tensors.linear, tensors.reshape)
    EXPORT_LINEAR_BIAS_F32(rt, x=tensors.layer_norm, weight=tensors.p_attn_k_proj_weight, bias=tensors.p_attn_k_proj_bias, output=tensors.linear_1)
    _alias(rt, tensors.linear_1, tensors.reshape_1)
    EXPORT_LINEAR_BIAS_F32(rt, x=tensors.layer_norm, weight=tensors.p_attn_v_proj_weight, bias=tensors.p_attn_v_proj_bias, output=tensors.linear_2)
    _alias(rt, tensors.linear_2, tensors.reshape_2)
    EXPORT_TRANSPOSE_F32_6A3397F037(rt, x=tensors.reshape, output=tensors.transpose)
    _alias(rt, tensors.transpose, tensors.unsqueeze)
    EXPORT_TRANSPOSE_F32_6A3397F037(rt, x=tensors.reshape_1, output=tensors.transpose_1)
    _alias(rt, tensors.transpose_1, tensors.unsqueeze_1)
    EXPORT_TRANSPOSE_F32_6A3397F037(rt, x=tensors.reshape_2, output=tensors.transpose_2)
    _alias(rt, tensors.transpose_2, tensors.unsqueeze_2)
    EXPORT_SDPA_MASKED_F32(rt, q=tensors.unsqueeze, k=tensors.unsqueeze_1, v=tensors.unsqueeze_2, mask=tensors.attention_mask, output=tensors.scaled_dot_product_attention)
    EXPORT_TRANSPOSE_F32_48D16B9B88(rt, x=tensors.scaled_dot_product_attention, output=tensors.transpose_3)
    _alias(rt, tensors.transpose_3, tensors.contiguous)
    _alias(rt, tensors.contiguous, tensors.reshape_3)
    EXPORT_LINEAR_BIAS_F32(rt, x=tensors.reshape_3, weight=tensors.p_attn_out_proj_weight, bias=tensors.p_attn_out_proj_bias, output=tensors.linear_3)
    ENCODER_LAYER_EXPORT_ADD_F32(rt, x=tensors.hidden_states, y=tensors.linear_3, output=tensors.add)
    EXPORT_LAYER_NORM_F32(rt, x=tensors.add, weight=tensors.p_final_layer_norm_weight, bias=tensors.p_final_layer_norm_bias, output=tensors.layer_norm_1)
    EXPORT_LINEAR_BIAS_F32_10(rt, x=tensors.layer_norm_1, weight=tensors.p_fc1_weight, bias=tensors.p_fc1_bias, output=tensors.linear_4)
    ENCODER_LAYER_EXPORT_GELU_F32(rt, x=tensors.linear_4, output=tensors.gelu)
    EXPORT_LINEAR_BIAS_F32_12(rt, x=tensors.gelu, weight=tensors.p_fc2_weight, bias=tensors.p_fc2_bias, output=tensors.linear_5)
    ENCODER_LAYER_EXPORT_ADD_F32(rt, x=tensors.add, y=tensors.linear_5, output=tensors.add_1)


def run_ln_post(rt: RuntimeSession, tensors: LnPostTensors) -> None:
    EXPORT_LAYER_NORM_F32(rt, x=tensors.input, weight=tensors.p_weight, bias=tensors.p_bias, output=tensors.layer_norm)


def run_proj1(rt: RuntimeSession, tensors: Proj1Tensors) -> None:
    EXPORT_LINEAR_BIAS_F32(rt, x=tensors.x, weight=tensors.p_weight, bias=tensors.p_bias, output=tensors.linear)
    PROJ1_EXPORT_GELU_F32(rt, x=tensors.linear, output=tensors.gelu)


def run_proj2(rt: RuntimeSession, tensors: Proj2Tensors) -> None:
    PROJ2_EXPORT_LINEAR_BIAS_F32(rt, x=tensors.input, weight=tensors.p_weight, bias=tensors.p_bias, output=tensors.linear)


def run_embed_tokens(rt: RuntimeSession, tensors: EmbedTokensTensors) -> None:
    EXPORT_EMBEDDING_F32(rt, weight=tensors.p_weight, indices=tensors.input, output=tensors.embedding)


def run_audio_inject(rt: RuntimeSession, tensors: AudioInjectTensors) -> None:
    _alias(rt, tensors.audio_features, tensors.unsqueeze)
    EXPORT_INDEX_COPY_F32_7BA4F1FF13(rt, cache=tensors.index_copy, index=tensors.audio_positions, src=tensors.unsqueeze)


def run_text_layer(rt: RuntimeSession, tensors: TextLayerTensors) -> None:
    _alias(rt, tensors.hidden_states, tensors.to)
    EXPORT_POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    EXPORT_MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    EXPORT_ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    EXPORT_RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    EXPORT_MUL_BROADCAST_LAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    _alias(rt, tensors.mul, tensors.to_1)
    EXPORT_MUL_LEFT_BROADCAST(rt, x=tensors.p_input_layernorm_weight, y=tensors.to_1, output=tensors.mul_1)
    TEXT_LAYER_EXPORT_LINEAR_NOBIAS_F32(rt, x=tensors.mul_1, weight=tensors.p_attn_q_proj_weight, output=tensors.linear)
    _alias(rt, tensors.linear, tensors.view)
    _alias(rt, tensors.view, tensors.to_2)
    EXPORT_POW_SCALAR_F32_7(rt, x=tensors.to_2, output=tensors.pow_2)
    EXPORT_MEAN_DIM_F32_8(rt, x=tensors.pow_2, output=tensors.mean_1)
    EXPORT_ADD_SCALAR_9(rt, x=tensors.mean_1, output=tensors.add_1)
    EXPORT_RSQRT_F32_10(rt, x=tensors.add_1, output=tensors.rsqrt_1)
    EXPORT_MUL_BROADCAST_LAST_11(rt, x=tensors.to_2, y=tensors.rsqrt_1, output=tensors.mul_2)
    _alias(rt, tensors.mul_2, tensors.to_3)
    EXPORT_MUL_LEFT_BROADCAST_12(rt, x=tensors.p_attn_q_norm_weight, y=tensors.to_3, output=tensors.mul_3)
    EXPORT_TRANSPOSE_F32_8E058A050E(rt, x=tensors.mul_3, output=tensors.transpose)
    EXPORT_LINEAR_NOBIAS_F32_14(rt, x=tensors.mul_1, weight=tensors.p_attn_k_proj_weight, output=tensors.linear_1)
    _alias(rt, tensors.linear_1, tensors.view_1)
    _alias(rt, tensors.view_1, tensors.to_4)
    EXPORT_POW_SCALAR_F32_15(rt, x=tensors.to_4, output=tensors.pow_3)
    EXPORT_MEAN_DIM_F32_16(rt, x=tensors.pow_3, output=tensors.mean_2)
    EXPORT_ADD_SCALAR_17(rt, x=tensors.mean_2, output=tensors.add_2)
    EXPORT_RSQRT_F32_18(rt, x=tensors.add_2, output=tensors.rsqrt_2)
    EXPORT_MUL_BROADCAST_LAST_19(rt, x=tensors.to_4, y=tensors.rsqrt_2, output=tensors.mul_4)
    _alias(rt, tensors.mul_4, tensors.to_5)
    EXPORT_MUL_LEFT_BROADCAST_20(rt, x=tensors.p_attn_k_norm_weight, y=tensors.to_5, output=tensors.mul_5)
    EXPORT_TRANSPOSE_F32_6392135058(rt, x=tensors.mul_5, output=tensors.transpose_1)
    EXPORT_LINEAR_NOBIAS_F32_22(rt, x=tensors.mul_1, weight=tensors.p_attn_v_proj_weight, output=tensors.linear_2)
    _alias(rt, tensors.linear_2, tensors.view_2)
    EXPORT_TRANSPOSE_F32_6392135058(rt, x=tensors.view_2, output=tensors.transpose_2)
    _alias(rt, tensors.position_embeddings_0, tensors.unsqueeze)
    _alias(rt, tensors.position_embeddings_1, tensors.unsqueeze_1)
    EXPORT_MUL_BROADCAST_INNER(rt, x=tensors.transpose, y=tensors.unsqueeze, output=tensors.mul_6)
    EXPORT_SLICE_F32(rt, x=tensors.transpose, output=tensors.slice_1)
    EXPORT_SLICE_F32_25(rt, x=tensors.transpose, output=tensors.slice_2)
    EXPORT_NEG_F32(rt, x=tensors.slice_2, output=tensors.neg)
    EXPORT_CAT_F32(rt, a=tensors.neg, b=tensors.slice_1, output=tensors.cat)
    EXPORT_MUL_BROADCAST_INNER(rt, x=tensors.cat, y=tensors.unsqueeze_1, output=tensors.mul_7)
    TEXT_LAYER_EXPORT_ADD_F32(rt, x=tensors.mul_6, y=tensors.mul_7, output=tensors.add_3)
    EXPORT_MUL_BROADCAST_INNER_29(rt, x=tensors.transpose_1, y=tensors.unsqueeze, output=tensors.mul_8)
    EXPORT_SLICE_F32_30(rt, x=tensors.transpose_1, output=tensors.slice_3)
    EXPORT_SLICE_F32_31(rt, x=tensors.transpose_1, output=tensors.slice_4)
    EXPORT_NEG_F32(rt, x=tensors.slice_4, output=tensors.neg_1)
    EXPORT_CAT_F32_32(rt, a=tensors.neg_1, b=tensors.slice_3, output=tensors.cat_1)
    EXPORT_MUL_BROADCAST_INNER_33(rt, x=tensors.cat_1, y=tensors.unsqueeze_1, output=tensors.mul_9)
    TEXT_LAYER_EXPORT_ADD_F32(rt, x=tensors.mul_8, y=tensors.mul_9, output=tensors.add_4)
    EXPORT_KV_CACHE_WRITE_F32(rt, cache=tensors.index_copy, cache_position=tensors.cache_position, src=tensors.add_4)
    EXPORT_KV_CACHE_WRITE_F32(rt, cache=tensors.index_copy_1, cache_position=tensors.cache_position, src=tensors.transpose_2)
    EXPORT_SDPA_CAUSAL_F32(rt, q=tensors.add_3, k=tensors.add_4, v=tensors.transpose_2, output=tensors.scaled_dot_product_attention)
    EXPORT_TRANSPOSE_F32_9E00CA2F33(rt, x=tensors.scaled_dot_product_attention, output=tensors.transpose_3)
    _alias(rt, tensors.transpose_3, tensors.contiguous)
    _alias(rt, tensors.contiguous, tensors.reshape)
    EXPORT_LINEAR_NOBIAS_F32_37(rt, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3)
    EXPORT_ADD_F32_38(rt, x=tensors.to, y=tensors.linear_3, output=tensors.add_5)
    _alias(rt, tensors.add_5, tensors.to_6)
    EXPORT_POW_SCALAR_F32(rt, x=tensors.to_6, output=tensors.pow_4)
    EXPORT_MEAN_DIM_F32(rt, x=tensors.pow_4, output=tensors.mean_3)
    EXPORT_ADD_SCALAR(rt, x=tensors.mean_3, output=tensors.add_6)
    EXPORT_RSQRT_F32(rt, x=tensors.add_6, output=tensors.rsqrt_3)
    EXPORT_MUL_BROADCAST_LAST(rt, x=tensors.to_6, y=tensors.rsqrt_3, output=tensors.mul_10)
    _alias(rt, tensors.mul_10, tensors.to_7)
    EXPORT_MUL_LEFT_BROADCAST(rt, x=tensors.p_post_attention_layernorm_weight, y=tensors.to_7, output=tensors.mul_11)
    EXPORT_LINEAR_NOBIAS_F32_39(rt, x=tensors.mul_11, weight=tensors.p_mlp_gate_proj_weight, output=tensors.linear_4)
    EXPORT_SILU_F32(rt, x=tensors.linear_4, output=tensors.silu)
    EXPORT_LINEAR_NOBIAS_F32_41(rt, x=tensors.mul_11, weight=tensors.p_mlp_up_proj_weight, output=tensors.linear_5)
    EXPORT_MUL_F32(rt, x=tensors.silu, y=tensors.linear_5, output=tensors.mul_12)
    EXPORT_LINEAR_NOBIAS_F32_43(rt, x=tensors.mul_12, weight=tensors.p_mlp_down_proj_weight, output=tensors.linear_6)
    EXPORT_ADD_F32_44(rt, x=tensors.to_6, y=tensors.linear_6, output=tensors.add_7)


def run_text_norm(rt: RuntimeSession, tensors: TextNormTensors) -> None:
    _alias(rt, tensors.hidden_states, tensors.to)
    EXPORT_POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    EXPORT_MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    EXPORT_ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    EXPORT_RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    EXPORT_MUL_BROADCAST_LAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    _alias(rt, tensors.mul, tensors.to_1)
    EXPORT_MUL_LEFT_BROADCAST(rt, x=tensors.p_weight, y=tensors.to_1, output=tensors.mul_1)


def run_lm_head(rt: RuntimeSession, tensors: LmHeadTensors) -> None:
    LM_HEAD_EXPORT_LINEAR_NOBIAS_F32(rt, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def run_decode_embed(rt: RuntimeSession, tensors: DecodeEmbedTensors) -> None:
    DECODE_EMBED_EXPORT_EMBEDDING_F32(rt, weight=tensors.p_weight, indices=tensors.input, output=tensors.embedding)


def run_decode_layer(rt: RuntimeSession, tensors: DecodeLayerTensors) -> None:
    _alias(rt, tensors.hidden_states, tensors.to)
    EXPORT_POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    DECODE_LAYER_EXPORT_MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    EXPORT_ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    EXPORT_RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    DECODE_LAYER_EXPORT_MUL_BROADCAST_LAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    _alias(rt, tensors.mul, tensors.to_1)
    EXPORT_MUL_LEFT_BROADCAST(rt, x=tensors.p_input_layernorm_weight, y=tensors.to_1, output=tensors.mul_1)
    DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32(rt, x=tensors.mul_1, weight=tensors.p_attn_q_proj_weight, output=tensors.linear)
    _alias(rt, tensors.linear, tensors.view)
    _alias(rt, tensors.view, tensors.to_2)
    EXPORT_POW_SCALAR_F32_7(rt, x=tensors.to_2, output=tensors.pow_2)
    DECODE_LAYER_EXPORT_MEAN_DIM_F32_8(rt, x=tensors.pow_2, output=tensors.mean_1)
    EXPORT_ADD_SCALAR_9(rt, x=tensors.mean_1, output=tensors.add_1)
    EXPORT_RSQRT_F32_10(rt, x=tensors.add_1, output=tensors.rsqrt_1)
    DECODE_LAYER_EXPORT_MUL_BROADCAST_LAST_11(rt, x=tensors.to_2, y=tensors.rsqrt_1, output=tensors.mul_2)
    _alias(rt, tensors.mul_2, tensors.to_3)
    EXPORT_MUL_LEFT_BROADCAST_12(rt, x=tensors.p_attn_q_norm_weight, y=tensors.to_3, output=tensors.mul_3)
    EXPORT_TRANSPOSE_F32_9884B7B82D(rt, x=tensors.mul_3, output=tensors.transpose)
    DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_14(rt, x=tensors.mul_1, weight=tensors.p_attn_k_proj_weight, output=tensors.linear_1)
    _alias(rt, tensors.linear_1, tensors.view_1)
    _alias(rt, tensors.view_1, tensors.to_4)
    EXPORT_POW_SCALAR_F32_15(rt, x=tensors.to_4, output=tensors.pow_3)
    DECODE_LAYER_EXPORT_MEAN_DIM_F32_16(rt, x=tensors.pow_3, output=tensors.mean_2)
    EXPORT_ADD_SCALAR_17(rt, x=tensors.mean_2, output=tensors.add_2)
    EXPORT_RSQRT_F32_18(rt, x=tensors.add_2, output=tensors.rsqrt_2)
    DECODE_LAYER_EXPORT_MUL_BROADCAST_LAST_19(rt, x=tensors.to_4, y=tensors.rsqrt_2, output=tensors.mul_4)
    _alias(rt, tensors.mul_4, tensors.to_5)
    EXPORT_MUL_LEFT_BROADCAST_20(rt, x=tensors.p_attn_k_norm_weight, y=tensors.to_5, output=tensors.mul_5)
    EXPORT_TRANSPOSE_F32_D509518B4F(rt, x=tensors.mul_5, output=tensors.transpose_1)
    DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_22(rt, x=tensors.mul_1, weight=tensors.p_attn_v_proj_weight, output=tensors.linear_2)
    _alias(rt, tensors.linear_2, tensors.view_2)
    EXPORT_TRANSPOSE_F32_D509518B4F(rt, x=tensors.view_2, output=tensors.transpose_2)
    _alias(rt, tensors.position_embeddings_0, tensors.unsqueeze)
    _alias(rt, tensors.position_embeddings_1, tensors.unsqueeze_1)
    DECODE_LAYER_EXPORT_MUL_BROADCAST_INNER(rt, x=tensors.transpose, y=tensors.unsqueeze, output=tensors.mul_6)
    DECODE_LAYER_EXPORT_SLICE_F32(rt, x=tensors.transpose, output=tensors.slice_1)
    DECODE_LAYER_EXPORT_SLICE_F32_25(rt, x=tensors.transpose, output=tensors.slice_2)
    EXPORT_NEG_F32(rt, x=tensors.slice_2, output=tensors.neg)
    DECODE_LAYER_EXPORT_CAT_F32(rt, a=tensors.neg, b=tensors.slice_1, output=tensors.cat)
    DECODE_LAYER_EXPORT_MUL_BROADCAST_INNER(rt, x=tensors.cat, y=tensors.unsqueeze_1, output=tensors.mul_7)
    DECODE_LAYER_EXPORT_ADD_F32(rt, x=tensors.mul_6, y=tensors.mul_7, output=tensors.add_3)
    DECODE_LAYER_EXPORT_MUL_BROADCAST_INNER_29(rt, x=tensors.transpose_1, y=tensors.unsqueeze, output=tensors.mul_8)
    DECODE_LAYER_EXPORT_SLICE_F32_30(rt, x=tensors.transpose_1, output=tensors.slice_3)
    DECODE_LAYER_EXPORT_SLICE_F32_31(rt, x=tensors.transpose_1, output=tensors.slice_4)
    EXPORT_NEG_F32(rt, x=tensors.slice_4, output=tensors.neg_1)
    DECODE_LAYER_EXPORT_CAT_F32_32(rt, a=tensors.neg_1, b=tensors.slice_3, output=tensors.cat_1)
    DECODE_LAYER_EXPORT_MUL_BROADCAST_INNER_33(rt, x=tensors.cat_1, y=tensors.unsqueeze_1, output=tensors.mul_9)
    DECODE_LAYER_EXPORT_ADD_F32(rt, x=tensors.mul_8, y=tensors.mul_9, output=tensors.add_4)
    EXPORT_KV_CACHE_WRITE_F32(rt, cache=tensors.index_copy, cache_position=tensors.cache_position, src=tensors.add_4)
    EXPORT_KV_CACHE_WRITE_F32(rt, cache=tensors.index_copy_1, cache_position=tensors.cache_position, src=tensors.transpose_2)
    EXPORT_SDPA_DECODE_CACHE_F32(rt, q=tensors.add_3, k=tensors.index_copy, v=tensors.index_copy_1, cache_position=tensors.cache_position, output=tensors.scaled_dot_product_attention)
    EXPORT_TRANSPOSE_F32_9E77B1CEE2(rt, x=tensors.scaled_dot_product_attention, output=tensors.transpose_3)
    _alias(rt, tensors.transpose_3, tensors.reshape)
    DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_37(rt, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3)
    EXPORT_ADD_F32_38(rt, x=tensors.to, y=tensors.linear_3, output=tensors.add_5)
    _alias(rt, tensors.add_5, tensors.to_6)
    EXPORT_POW_SCALAR_F32(rt, x=tensors.to_6, output=tensors.pow_4)
    DECODE_LAYER_EXPORT_MEAN_DIM_F32(rt, x=tensors.pow_4, output=tensors.mean_3)
    EXPORT_ADD_SCALAR(rt, x=tensors.mean_3, output=tensors.add_6)
    EXPORT_RSQRT_F32(rt, x=tensors.add_6, output=tensors.rsqrt_3)
    DECODE_LAYER_EXPORT_MUL_BROADCAST_LAST(rt, x=tensors.to_6, y=tensors.rsqrt_3, output=tensors.mul_10)
    _alias(rt, tensors.mul_10, tensors.to_7)
    EXPORT_MUL_LEFT_BROADCAST(rt, x=tensors.p_post_attention_layernorm_weight, y=tensors.to_7, output=tensors.mul_11)
    DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_39(rt, x=tensors.mul_11, weight=tensors.p_mlp_gate_proj_weight, output=tensors.linear_4)
    EXPORT_SILU_F32(rt, x=tensors.linear_4, output=tensors.silu)
    DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_41(rt, x=tensors.mul_11, weight=tensors.p_mlp_up_proj_weight, output=tensors.linear_5)
    EXPORT_MUL_F32(rt, x=tensors.silu, y=tensors.linear_5, output=tensors.mul_12)
    DECODE_LAYER_EXPORT_LINEAR_NOBIAS_F32_43(rt, x=tensors.mul_12, weight=tensors.p_mlp_down_proj_weight, output=tensors.linear_6)
    EXPORT_ADD_F32_44(rt, x=tensors.to_6, y=tensors.linear_6, output=tensors.add_7)


def run_decode_norm(rt: RuntimeSession, tensors: DecodeNormTensors) -> None:
    _alias(rt, tensors.hidden_states, tensors.to)
    EXPORT_POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    DECODE_NORM_EXPORT_MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    EXPORT_ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    EXPORT_RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    DECODE_NORM_EXPORT_MUL_BROADCAST_LAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    _alias(rt, tensors.mul, tensors.to_1)
    EXPORT_MUL_LEFT_BROADCAST(rt, x=tensors.p_weight, y=tensors.to_1, output=tensors.mul_1)


def run_decode_lm_head(rt: RuntimeSession, tensors: DecodeLmHeadTensors) -> None:
    DECODE_LM_HEAD_EXPORT_LINEAR_NOBIAS_F32(rt, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def decode_step_inputs(
    *,
    decode_embed_t: DecodeEmbedTensors,
    decode_layer_ts: Sequence[DecodeLayerTensors],
    eos_token_ids: LogicalTensor,
    token_index: LogicalTensor,
    token: int,
    cache_position: int,
    eos_token_array: np.ndarray,
    token_index_value: int,
) -> dict[LogicalTensor, np.ndarray]:
    if not decode_layer_ts:
        raise ValueError("decode_layer_ts must not be empty")
    return {
        decode_embed_t.input: np.array([[token]], dtype=np.int64),
        decode_layer_ts[0].cache_position: np.array([cache_position], dtype=np.int64),
        eos_token_ids: np.ascontiguousarray(eos_token_array, dtype=np.int64),
        token_index: np.array([token_index_value], dtype=np.int64),
    }


def run_decode_step(
    rt: RuntimeSession,
    *,
    decode_embed_t: DecodeEmbedTensors,
    decode_layer_ts: Sequence[DecodeLayerTensors],
    decode_norm_t: DecodeNormTensors,
    decode_lm_head_t: DecodeLmHeadTensors,
    eos_token_ids: LogicalTensor,
    next_token: LogicalTensor,
    done: LogicalTensor,
    token_index: LogicalTensor,
    generated_tokens: LogicalTensor,
    generated_length: LogicalTensor,
    stopped: LogicalTensor,
    step: int,
) -> int:
    if not decode_layer_ts:
        raise ValueError("decode_layer_ts must not be empty")
    with rt.frame(f"spike.decode.{step:04d}"):
        run_decode_embed(rt, decode_embed_t)
        for layer_tensors in decode_layer_ts:
            run_decode_layer(rt, layer_tensors)
        run_decode_norm(rt, decode_norm_t)
        run_decode_lm_head(rt, decode_lm_head_t)
        QWEN3_ASR_TOKEN_SELECT_GREEDY_F32(
            rt,
            logits=decode_lm_head_t.linear,
            eos_token_ids=eos_token_ids,
            next_token=next_token,
            done=done,
        )
        QWEN3_ASR_TOKEN_STORE_EOS_F32(
            rt,
            next_token=next_token,
            token_index=token_index,
            done=done,
            generated_tokens=generated_tokens,
            generated_length=generated_length,
            stopped=stopped,
        )
    return int(rt.read_request_state(next_token).reshape(-1)[0])


def _alias(rt: RuntimeSession, src: LogicalTensor, dst: LogicalTensor) -> None:
    rt._materialize_read(src)
    with dst.runtime_write_scope():
        dst.buffer = src.buffer
        dst.descriptor_nbytes = src.descriptor_nbytes
        dst.version = src.version
        dst.writer = src.writer
    rt._current_frame().written_tensors.append(dst)
