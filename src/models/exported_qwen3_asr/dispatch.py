"""Generated dispatch functions for all submodules."""

from __future__ import annotations

import numpy as np

from models.exported_qwen3_asr.tensors.model import model_tensors
from models.optimized_qwen3_asr.shaders.token_select_f32 import QWEN3_ASR_TOKEN_SELECT_GREEDY_F32
from models.optimized_qwen3_asr.shaders.token_store_f32 import QWEN3_ASR_TOKEN_STORE_EOS_F32
from torch2vk.runtime.rope_table import run_rope_table_f32
from models.exported_qwen3_asr.shaders.add_f32 import ADD_F32
from models.exported_qwen3_asr.shaders.add_f32_17 import ADD_F32_17
from models.exported_qwen3_asr.shaders.add_f32_21 import ADD_F32_21
from models.exported_qwen3_asr.shaders.add_f32_38 import ADD_F32_38
from models.exported_qwen3_asr.shaders.add_f32_44 import ADD_F32_44
from models.exported_qwen3_asr.shaders.add_scalar import ADD_SCALAR
from models.exported_qwen3_asr.shaders.add_scalar_17 import ADD_SCALAR_17
from models.exported_qwen3_asr.shaders.add_scalar_9 import ADD_SCALAR_9
from models.exported_qwen3_asr.shaders.cat_f32 import CAT_F32
from models.exported_qwen3_asr.shaders.cat_f32_32 import CAT_F32_32
from models.exported_qwen3_asr.shaders.conv2d_f32 import CONV2D_F32
from models.exported_qwen3_asr.shaders.conv2d_f32_2 import CONV2D_F32_2
from models.exported_qwen3_asr.shaders.conv2d_f32_3 import CONV2D_F32_3
from models.exported_qwen3_asr.shaders.decode_embed_embedding_f32 import DECODE_EMBED_EMBEDDING_F32
from models.exported_qwen3_asr.shaders.decode_layer_add_f32 import DECODE_LAYER_ADD_F32
from models.exported_qwen3_asr.shaders.decode_layer_cat_f32 import DECODE_LAYER_CAT_F32
from models.exported_qwen3_asr.shaders.decode_layer_cat_f32_32 import DECODE_LAYER_CAT_F32_32
from models.exported_qwen3_asr.shaders.decode_layer_linear_nobias_f32 import DECODE_LAYER_LINEAR_NOBIAS_F32
from models.exported_qwen3_asr.shaders.decode_layer_linear_nobias_f32_14 import DECODE_LAYER_LINEAR_NOBIAS_F32_14
from models.exported_qwen3_asr.shaders.decode_layer_linear_nobias_f32_22 import DECODE_LAYER_LINEAR_NOBIAS_F32_22
from models.exported_qwen3_asr.shaders.decode_layer_linear_nobias_f32_37 import DECODE_LAYER_LINEAR_NOBIAS_F32_37
from models.exported_qwen3_asr.shaders.decode_layer_linear_nobias_f32_39 import DECODE_LAYER_LINEAR_NOBIAS_F32_39
from models.exported_qwen3_asr.shaders.decode_layer_linear_nobias_f32_41 import DECODE_LAYER_LINEAR_NOBIAS_F32_41
from models.exported_qwen3_asr.shaders.decode_layer_linear_nobias_f32_43 import DECODE_LAYER_LINEAR_NOBIAS_F32_43
from models.exported_qwen3_asr.shaders.decode_layer_mean_dim_f32 import DECODE_LAYER_MEAN_DIM_F32
from models.exported_qwen3_asr.shaders.decode_layer_mean_dim_f32_16 import DECODE_LAYER_MEAN_DIM_F32_16
from models.exported_qwen3_asr.shaders.decode_layer_mean_dim_f32_8 import DECODE_LAYER_MEAN_DIM_F32_8
from models.exported_qwen3_asr.shaders.decode_layer_mul_broadcast_inner import DECODE_LAYER_MUL_BROADCAST_INNER
from models.exported_qwen3_asr.shaders.decode_layer_mul_broadcast_inner_29 import DECODE_LAYER_MUL_BROADCAST_INNER_29
from models.exported_qwen3_asr.shaders.decode_layer_mul_broadcast_inner_33 import DECODE_LAYER_MUL_BROADCAST_INNER_33
from models.exported_qwen3_asr.shaders.decode_layer_mul_broadcast_last import DECODE_LAYER_MUL_BROADCAST_LAST
from models.exported_qwen3_asr.shaders.decode_layer_mul_broadcast_last_11 import DECODE_LAYER_MUL_BROADCAST_LAST_11
from models.exported_qwen3_asr.shaders.decode_layer_mul_broadcast_last_19 import DECODE_LAYER_MUL_BROADCAST_LAST_19
from models.exported_qwen3_asr.shaders.decode_layer_slice_f32 import DECODE_LAYER_SLICE_F32
from models.exported_qwen3_asr.shaders.decode_layer_slice_f32_25 import DECODE_LAYER_SLICE_F32_25
from models.exported_qwen3_asr.shaders.decode_layer_slice_f32_30 import DECODE_LAYER_SLICE_F32_30
from models.exported_qwen3_asr.shaders.decode_layer_slice_f32_31 import DECODE_LAYER_SLICE_F32_31
from models.exported_qwen3_asr.shaders.decode_lm_head_linear_nobias_f32 import DECODE_LM_HEAD_LINEAR_NOBIAS_F32
from models.exported_qwen3_asr.shaders.decode_norm_mean_dim_f32 import DECODE_NORM_MEAN_DIM_F32
from models.exported_qwen3_asr.shaders.decode_norm_mul_broadcast_last import DECODE_NORM_MUL_BROADCAST_LAST
from models.exported_qwen3_asr.shaders.embedding_f32 import EMBEDDING_F32
from models.exported_qwen3_asr.shaders.gelu_f32 import GELU_F32
from models.exported_qwen3_asr.shaders.gelu_f32_124 import GELU_F32_124
from models.exported_qwen3_asr.shaders.gelu_f32_19 import GELU_F32_19
from models.exported_qwen3_asr.shaders.index_copy_f32_7ba4f1ff13 import INDEX_COPY_F32_7BA4F1FF13
from models.exported_qwen3_asr.shaders.index_select_f32_c6680f8d95 import INDEX_SELECT_F32_C6680F8D95
from models.exported_qwen3_asr.shaders.kv_cache_write_f32 import KV_CACHE_WRITE_F32
from models.exported_qwen3_asr.shaders.layer_norm_f32 import LAYER_NORM_F32
from models.exported_qwen3_asr.shaders.linear_bias_f32 import LINEAR_BIAS_F32
from models.exported_qwen3_asr.shaders.linear_bias_f32_125 import LINEAR_BIAS_F32_125
from models.exported_qwen3_asr.shaders.linear_bias_f32_18 import LINEAR_BIAS_F32_18
from models.exported_qwen3_asr.shaders.linear_bias_f32_20 import LINEAR_BIAS_F32_20
from models.exported_qwen3_asr.shaders.linear_nobias_f32 import LINEAR_NOBIAS_F32
from models.exported_qwen3_asr.shaders.linear_nobias_f32_14 import LINEAR_NOBIAS_F32_14
from models.exported_qwen3_asr.shaders.linear_nobias_f32_22 import LINEAR_NOBIAS_F32_22
from models.exported_qwen3_asr.shaders.linear_nobias_f32_37 import LINEAR_NOBIAS_F32_37
from models.exported_qwen3_asr.shaders.linear_nobias_f32_39 import LINEAR_NOBIAS_F32_39
from models.exported_qwen3_asr.shaders.linear_nobias_f32_41 import LINEAR_NOBIAS_F32_41
from models.exported_qwen3_asr.shaders.linear_nobias_f32_43 import LINEAR_NOBIAS_F32_43
from models.exported_qwen3_asr.shaders.lm_head_linear_nobias_f32 import LM_HEAD_LINEAR_NOBIAS_F32
from models.exported_qwen3_asr.shaders.mean_dim_f32 import MEAN_DIM_F32
from models.exported_qwen3_asr.shaders.mean_dim_f32_16 import MEAN_DIM_F32_16
from models.exported_qwen3_asr.shaders.mean_dim_f32_8 import MEAN_DIM_F32_8
from models.exported_qwen3_asr.shaders.mul_broadcast_inner import MUL_BROADCAST_INNER
from models.exported_qwen3_asr.shaders.mul_broadcast_inner_29 import MUL_BROADCAST_INNER_29
from models.exported_qwen3_asr.shaders.mul_broadcast_inner_33 import MUL_BROADCAST_INNER_33
from models.exported_qwen3_asr.shaders.mul_broadcast_last import MUL_BROADCAST_LAST
from models.exported_qwen3_asr.shaders.mul_broadcast_last_11 import MUL_BROADCAST_LAST_11
from models.exported_qwen3_asr.shaders.mul_broadcast_last_19 import MUL_BROADCAST_LAST_19
from models.exported_qwen3_asr.shaders.mul_f32 import MUL_F32
from models.exported_qwen3_asr.shaders.mul_left_broadcast import MUL_LEFT_BROADCAST
from models.exported_qwen3_asr.shaders.mul_left_broadcast_12 import MUL_LEFT_BROADCAST_12
from models.exported_qwen3_asr.shaders.mul_left_broadcast_20 import MUL_LEFT_BROADCAST_20
from models.exported_qwen3_asr.shaders.neg_f32 import NEG_F32
from models.exported_qwen3_asr.shaders.pow_scalar_f32 import POW_SCALAR_F32
from models.exported_qwen3_asr.shaders.pow_scalar_f32_15 import POW_SCALAR_F32_15
from models.exported_qwen3_asr.shaders.pow_scalar_f32_7 import POW_SCALAR_F32_7
from models.exported_qwen3_asr.shaders.rsqrt_f32 import RSQRT_F32
from models.exported_qwen3_asr.shaders.rsqrt_f32_10 import RSQRT_F32_10
from models.exported_qwen3_asr.shaders.rsqrt_f32_18 import RSQRT_F32_18
from models.exported_qwen3_asr.shaders.sdpa_causal_f32 import SDPA_CAUSAL_F32
from models.exported_qwen3_asr.shaders.sdpa_decode_cache_f32 import SDPA_DECODE_CACHE_F32
from models.exported_qwen3_asr.shaders.sdpa_masked_f32 import SDPA_MASKED_F32
from models.exported_qwen3_asr.shaders.silu_f32 import SILU_F32
from models.exported_qwen3_asr.shaders.slice_f32 import SLICE_F32
from models.exported_qwen3_asr.shaders.slice_f32_25 import SLICE_F32_25
from models.exported_qwen3_asr.shaders.slice_f32_30 import SLICE_F32_30
from models.exported_qwen3_asr.shaders.slice_f32_31 import SLICE_F32_31
from models.exported_qwen3_asr.shaders.text_layer_add_f32 import TEXT_LAYER_ADD_F32
from models.exported_qwen3_asr.shaders.text_layer_linear_nobias_f32 import TEXT_LAYER_LINEAR_NOBIAS_F32
from models.exported_qwen3_asr.shaders.transpose_f32_48d16b9b88 import TRANSPOSE_F32_48D16B9B88
from models.exported_qwen3_asr.shaders.transpose_f32_6392135058 import TRANSPOSE_F32_6392135058
from models.exported_qwen3_asr.shaders.transpose_f32_6a3397f037 import TRANSPOSE_F32_6A3397F037
from models.exported_qwen3_asr.shaders.transpose_f32_8e058a050e import TRANSPOSE_F32_8E058A050E
from models.exported_qwen3_asr.shaders.transpose_f32_9884b7b82d import TRANSPOSE_F32_9884B7B82D
from models.exported_qwen3_asr.shaders.transpose_f32_9e00ca2f33 import TRANSPOSE_F32_9E00CA2F33
from models.exported_qwen3_asr.shaders.transpose_f32_9e77b1cee2 import TRANSPOSE_F32_9E77B1CEE2
from models.exported_qwen3_asr.shaders.transpose_f32_d509518b4f import TRANSPOSE_F32_D509518B4F
from models.exported_qwen3_asr.shaders.transpose_f32_d95ce920ac import TRANSPOSE_F32_D95CE920AC
from models.exported_qwen3_asr.tensors.audio_encoder import AudioEncoderTensors
from models.exported_qwen3_asr.tensors.audio_inject import AudioInjectTensors
from models.exported_qwen3_asr.tensors.decode_embed import DecodeEmbedTensors
from models.exported_qwen3_asr.tensors.decode_layer import DecodeLayerTensors
from models.exported_qwen3_asr.tensors.decode_lm_head import DecodeLmHeadTensors
from models.exported_qwen3_asr.tensors.decode_norm import DecodeNormTensors
from models.exported_qwen3_asr.tensors.embed_tokens import EmbedTokensTensors
from models.exported_qwen3_asr.tensors.lm_head import LmHeadTensors
from models.exported_qwen3_asr.tensors.text_layer import TextLayerTensors
from models.exported_qwen3_asr.tensors.text_norm import TextNormTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession


def _run_audio_encoder_with_tensors(rt: RuntimeSession, tensors: AudioEncoderTensors) -> None:
    CONV2D_F32(rt, x=tensors.x, weight=tensors.p_audio_tower_conv2d1_weight, bias=tensors.p_audio_tower_conv2d1_bias, output=tensors.conv2d)
    GELU_F32(rt, x=tensors.conv2d, output=tensors.gelu)
    CONV2D_F32_2(rt, x=tensors.gelu, weight=tensors.p_audio_tower_conv2d2_weight, bias=tensors.p_audio_tower_conv2d2_bias, output=tensors.conv2d_1)
    GELU_F32(rt, x=tensors.conv2d_1, output=tensors.gelu_1)
    CONV2D_F32_3(rt, x=tensors.gelu_1, weight=tensors.p_audio_tower_conv2d3_weight, bias=tensors.p_audio_tower_conv2d3_bias, output=tensors.conv2d_2)
    GELU_F32(rt, x=tensors.conv2d_2, output=tensors.gelu_2)
    TRANSPOSE_F32_D95CE920AC(rt, x=tensors.reshape, output=tensors.transpose)
    LINEAR_NOBIAS_F32(rt, x=tensors.transpose, weight=tensors.p_audio_tower_conv_out_weight, output=tensors.linear)
    ADD_F32(rt, x=tensors.linear, y=tensors.position_embedding, output=tensors.add)
    INDEX_SELECT_F32_C6680F8D95(rt, x=tensors.reshape_1, index=tensors.compact_index, output=tensors.index_select)
    carry = tensors.index_select
    for layer_t in tensors.layers:
        LAYER_NORM_F32(rt, x=carry, weight=layer_t.p_audio_tower_layers_0_self_attn_layer_norm_weight, bias=layer_t.p_audio_tower_layers_0_self_attn_layer_norm_bias, output=layer_t.layer_norm)
        LINEAR_BIAS_F32(rt, x=layer_t.layer_norm, weight=layer_t.p_audio_tower_layers_0_self_attn_q_proj_weight, bias=layer_t.p_audio_tower_layers_0_self_attn_q_proj_bias, output=layer_t.linear_1)
        LINEAR_BIAS_F32(rt, x=layer_t.layer_norm, weight=layer_t.p_audio_tower_layers_0_self_attn_k_proj_weight, bias=layer_t.p_audio_tower_layers_0_self_attn_k_proj_bias, output=layer_t.linear_2)
        LINEAR_BIAS_F32(rt, x=layer_t.layer_norm, weight=layer_t.p_audio_tower_layers_0_self_attn_v_proj_weight, bias=layer_t.p_audio_tower_layers_0_self_attn_v_proj_bias, output=layer_t.linear_3)
        TRANSPOSE_F32_6A3397F037(rt, x=layer_t.reshape_2, output=layer_t.transpose_1)
        TRANSPOSE_F32_6A3397F037(rt, x=layer_t.reshape_3, output=layer_t.transpose_2)
        TRANSPOSE_F32_6A3397F037(rt, x=layer_t.reshape_4, output=layer_t.transpose_3)
        SDPA_MASKED_F32(rt, q=layer_t.unsqueeze, k=layer_t.unsqueeze_1, v=layer_t.unsqueeze_2, mask=tensors.attention_mask, output=layer_t.scaled_dot_product_attention)
        TRANSPOSE_F32_48D16B9B88(rt, x=layer_t.scaled_dot_product_attention, output=layer_t.transpose_4)
        LINEAR_BIAS_F32(rt, x=layer_t.reshape_5, weight=layer_t.p_audio_tower_layers_0_self_attn_out_proj_weight, bias=layer_t.p_audio_tower_layers_0_self_attn_out_proj_bias, output=layer_t.linear_4)
        ADD_F32_17(rt, x=carry, y=layer_t.linear_4, output=layer_t.add_1)
        LAYER_NORM_F32(rt, x=layer_t.add_1, weight=layer_t.p_audio_tower_layers_0_final_layer_norm_weight, bias=layer_t.p_audio_tower_layers_0_final_layer_norm_bias, output=layer_t.layer_norm_1)
        LINEAR_BIAS_F32_18(rt, x=layer_t.layer_norm_1, weight=layer_t.p_audio_tower_layers_0_fc1_weight, bias=layer_t.p_audio_tower_layers_0_fc1_bias, output=layer_t.linear_5)
        GELU_F32_19(rt, x=layer_t.linear_5, output=layer_t.gelu_3)
        LINEAR_BIAS_F32_20(rt, x=layer_t.gelu_3, weight=layer_t.p_audio_tower_layers_0_fc2_weight, bias=layer_t.p_audio_tower_layers_0_fc2_bias, output=layer_t.linear_6)
        ADD_F32_21(rt, x=layer_t.add_1, y=layer_t.linear_6, output=layer_t.add_2)
        carry = layer_t.add_2
    LAYER_NORM_F32(rt, x=carry, weight=tensors.p_audio_tower_ln_post_weight, bias=tensors.p_audio_tower_ln_post_bias, output=tensors.layer_norm_36)
    LINEAR_BIAS_F32(rt, x=tensors.layer_norm_36, weight=tensors.p_audio_tower_proj1_weight, bias=tensors.p_audio_tower_proj1_bias, output=tensors.linear_109)
    GELU_F32_124(rt, x=tensors.linear_109, output=tensors.gelu_21)
    LINEAR_BIAS_F32_125(rt, x=tensors.gelu_21, weight=tensors.p_audio_tower_proj2_weight, bias=tensors.p_audio_tower_proj2_bias, output=tensors.linear_110)


def _run_embed_tokens_with_tensors(rt: RuntimeSession, tensors: EmbedTokensTensors) -> None:
    EMBEDDING_F32(rt, weight=tensors.p_weight, indices=tensors.input, output=tensors.embedding)


def _run_audio_inject_with_tensors(rt: RuntimeSession, tensors: AudioInjectTensors) -> None:
    INDEX_COPY_F32_7BA4F1FF13(rt, cache=tensors.index_copy, index=tensors.audio_positions, src=tensors.unsqueeze)


def _run_text_layer_with_tensors(rt: RuntimeSession, tensors: TextLayerTensors) -> None:
    POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    MUL_BROADCAST_LAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    MUL_LEFT_BROADCAST(rt, x=tensors.p_input_layernorm_weight, y=tensors.to_1, output=tensors.mul_1)
    TEXT_LAYER_LINEAR_NOBIAS_F32(rt, x=tensors.mul_1, weight=tensors.p_attn_q_proj_weight, output=tensors.linear)
    POW_SCALAR_F32_7(rt, x=tensors.to_2, output=tensors.pow_2)
    MEAN_DIM_F32_8(rt, x=tensors.pow_2, output=tensors.mean_1)
    ADD_SCALAR_9(rt, x=tensors.mean_1, output=tensors.add_1)
    RSQRT_F32_10(rt, x=tensors.add_1, output=tensors.rsqrt_1)
    MUL_BROADCAST_LAST_11(rt, x=tensors.to_2, y=tensors.rsqrt_1, output=tensors.mul_2)
    MUL_LEFT_BROADCAST_12(rt, x=tensors.p_attn_q_norm_weight, y=tensors.to_3, output=tensors.mul_3)
    TRANSPOSE_F32_8E058A050E(rt, x=tensors.mul_3, output=tensors.transpose)
    LINEAR_NOBIAS_F32_14(rt, x=tensors.mul_1, weight=tensors.p_attn_k_proj_weight, output=tensors.linear_1)
    POW_SCALAR_F32_15(rt, x=tensors.to_4, output=tensors.pow_3)
    MEAN_DIM_F32_16(rt, x=tensors.pow_3, output=tensors.mean_2)
    ADD_SCALAR_17(rt, x=tensors.mean_2, output=tensors.add_2)
    RSQRT_F32_18(rt, x=tensors.add_2, output=tensors.rsqrt_2)
    MUL_BROADCAST_LAST_19(rt, x=tensors.to_4, y=tensors.rsqrt_2, output=tensors.mul_4)
    MUL_LEFT_BROADCAST_20(rt, x=tensors.p_attn_k_norm_weight, y=tensors.to_5, output=tensors.mul_5)
    TRANSPOSE_F32_6392135058(rt, x=tensors.mul_5, output=tensors.transpose_1)
    LINEAR_NOBIAS_F32_22(rt, x=tensors.mul_1, weight=tensors.p_attn_v_proj_weight, output=tensors.linear_2)
    TRANSPOSE_F32_6392135058(rt, x=tensors.view_2, output=tensors.transpose_2)
    MUL_BROADCAST_INNER(rt, x=tensors.transpose, y=tensors.unsqueeze, output=tensors.mul_6)
    SLICE_F32(rt, x=tensors.transpose, output=tensors.slice_1)
    SLICE_F32_25(rt, x=tensors.transpose, output=tensors.slice_2)
    NEG_F32(rt, x=tensors.slice_2, output=tensors.neg)
    CAT_F32(rt, a=tensors.neg, b=tensors.slice_1, output=tensors.cat)
    MUL_BROADCAST_INNER(rt, x=tensors.cat, y=tensors.unsqueeze_1, output=tensors.mul_7)
    TEXT_LAYER_ADD_F32(rt, x=tensors.mul_6, y=tensors.mul_7, output=tensors.add_3)
    MUL_BROADCAST_INNER_29(rt, x=tensors.transpose_1, y=tensors.unsqueeze, output=tensors.mul_8)
    SLICE_F32_30(rt, x=tensors.transpose_1, output=tensors.slice_3)
    SLICE_F32_31(rt, x=tensors.transpose_1, output=tensors.slice_4)
    NEG_F32(rt, x=tensors.slice_4, output=tensors.neg_1)
    CAT_F32_32(rt, a=tensors.neg_1, b=tensors.slice_3, output=tensors.cat_1)
    MUL_BROADCAST_INNER_33(rt, x=tensors.cat_1, y=tensors.unsqueeze_1, output=tensors.mul_9)
    TEXT_LAYER_ADD_F32(rt, x=tensors.mul_8, y=tensors.mul_9, output=tensors.add_4)
    KV_CACHE_WRITE_F32(rt, cache=tensors.index_copy, cache_position=tensors.cache_position, src=tensors.add_4)
    KV_CACHE_WRITE_F32(rt, cache=tensors.index_copy_1, cache_position=tensors.cache_position, src=tensors.transpose_2)
    SDPA_CAUSAL_F32(rt, q=tensors.add_3, k=tensors.add_4, v=tensors.transpose_2, output=tensors.scaled_dot_product_attention)
    TRANSPOSE_F32_9E00CA2F33(rt, x=tensors.scaled_dot_product_attention, output=tensors.transpose_3)
    LINEAR_NOBIAS_F32_37(rt, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3)
    ADD_F32_38(rt, x=tensors.to, y=tensors.linear_3, output=tensors.add_5)
    POW_SCALAR_F32(rt, x=tensors.to_6, output=tensors.pow_4)
    MEAN_DIM_F32(rt, x=tensors.pow_4, output=tensors.mean_3)
    ADD_SCALAR(rt, x=tensors.mean_3, output=tensors.add_6)
    RSQRT_F32(rt, x=tensors.add_6, output=tensors.rsqrt_3)
    MUL_BROADCAST_LAST(rt, x=tensors.to_6, y=tensors.rsqrt_3, output=tensors.mul_10)
    MUL_LEFT_BROADCAST(rt, x=tensors.p_post_attention_layernorm_weight, y=tensors.to_7, output=tensors.mul_11)
    LINEAR_NOBIAS_F32_39(rt, x=tensors.mul_11, weight=tensors.p_mlp_gate_proj_weight, output=tensors.linear_4)
    SILU_F32(rt, x=tensors.linear_4, output=tensors.silu)
    LINEAR_NOBIAS_F32_41(rt, x=tensors.mul_11, weight=tensors.p_mlp_up_proj_weight, output=tensors.linear_5)
    MUL_F32(rt, x=tensors.silu, y=tensors.linear_5, output=tensors.mul_12)
    LINEAR_NOBIAS_F32_43(rt, x=tensors.mul_12, weight=tensors.p_mlp_down_proj_weight, output=tensors.linear_6)
    ADD_F32_44(rt, x=tensors.to_6, y=tensors.linear_6, output=tensors.add_7)


def _run_text_norm_with_tensors(rt: RuntimeSession, tensors: TextNormTensors) -> None:
    POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    MUL_BROADCAST_LAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    MUL_LEFT_BROADCAST(rt, x=tensors.p_weight, y=tensors.to_1, output=tensors.mul_1)


def _run_lm_head_with_tensors(rt: RuntimeSession, tensors: LmHeadTensors) -> None:
    LM_HEAD_LINEAR_NOBIAS_F32(rt, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def _run_decode_embed_with_tensors(rt: RuntimeSession, tensors: DecodeEmbedTensors) -> None:
    DECODE_EMBED_EMBEDDING_F32(rt, weight=tensors.p_weight, indices=tensors.input, output=tensors.embedding)


def _run_decode_layer_with_tensors(rt: RuntimeSession, tensors: DecodeLayerTensors) -> None:
    POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    DECODE_LAYER_MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    DECODE_LAYER_MUL_BROADCAST_LAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    MUL_LEFT_BROADCAST(rt, x=tensors.p_input_layernorm_weight, y=tensors.to_1, output=tensors.mul_1)
    DECODE_LAYER_LINEAR_NOBIAS_F32(rt, x=tensors.mul_1, weight=tensors.p_attn_q_proj_weight, output=tensors.linear)
    POW_SCALAR_F32_7(rt, x=tensors.to_2, output=tensors.pow_2)
    DECODE_LAYER_MEAN_DIM_F32_8(rt, x=tensors.pow_2, output=tensors.mean_1)
    ADD_SCALAR_9(rt, x=tensors.mean_1, output=tensors.add_1)
    RSQRT_F32_10(rt, x=tensors.add_1, output=tensors.rsqrt_1)
    DECODE_LAYER_MUL_BROADCAST_LAST_11(rt, x=tensors.to_2, y=tensors.rsqrt_1, output=tensors.mul_2)
    MUL_LEFT_BROADCAST_12(rt, x=tensors.p_attn_q_norm_weight, y=tensors.to_3, output=tensors.mul_3)
    TRANSPOSE_F32_9884B7B82D(rt, x=tensors.mul_3, output=tensors.transpose)
    DECODE_LAYER_LINEAR_NOBIAS_F32_14(rt, x=tensors.mul_1, weight=tensors.p_attn_k_proj_weight, output=tensors.linear_1)
    POW_SCALAR_F32_15(rt, x=tensors.to_4, output=tensors.pow_3)
    DECODE_LAYER_MEAN_DIM_F32_16(rt, x=tensors.pow_3, output=tensors.mean_2)
    ADD_SCALAR_17(rt, x=tensors.mean_2, output=tensors.add_2)
    RSQRT_F32_18(rt, x=tensors.add_2, output=tensors.rsqrt_2)
    DECODE_LAYER_MUL_BROADCAST_LAST_19(rt, x=tensors.to_4, y=tensors.rsqrt_2, output=tensors.mul_4)
    MUL_LEFT_BROADCAST_20(rt, x=tensors.p_attn_k_norm_weight, y=tensors.to_5, output=tensors.mul_5)
    TRANSPOSE_F32_D509518B4F(rt, x=tensors.mul_5, output=tensors.transpose_1)
    DECODE_LAYER_LINEAR_NOBIAS_F32_22(rt, x=tensors.mul_1, weight=tensors.p_attn_v_proj_weight, output=tensors.linear_2)
    TRANSPOSE_F32_D509518B4F(rt, x=tensors.view_2, output=tensors.transpose_2)
    DECODE_LAYER_MUL_BROADCAST_INNER(rt, x=tensors.transpose, y=tensors.unsqueeze, output=tensors.mul_6)
    DECODE_LAYER_SLICE_F32(rt, x=tensors.transpose, output=tensors.slice_1)
    DECODE_LAYER_SLICE_F32_25(rt, x=tensors.transpose, output=tensors.slice_2)
    NEG_F32(rt, x=tensors.slice_2, output=tensors.neg)
    DECODE_LAYER_CAT_F32(rt, a=tensors.neg, b=tensors.slice_1, output=tensors.cat)
    DECODE_LAYER_MUL_BROADCAST_INNER(rt, x=tensors.cat, y=tensors.unsqueeze_1, output=tensors.mul_7)
    DECODE_LAYER_ADD_F32(rt, x=tensors.mul_6, y=tensors.mul_7, output=tensors.add_3)
    DECODE_LAYER_MUL_BROADCAST_INNER_29(rt, x=tensors.transpose_1, y=tensors.unsqueeze, output=tensors.mul_8)
    DECODE_LAYER_SLICE_F32_30(rt, x=tensors.transpose_1, output=tensors.slice_3)
    DECODE_LAYER_SLICE_F32_31(rt, x=tensors.transpose_1, output=tensors.slice_4)
    NEG_F32(rt, x=tensors.slice_4, output=tensors.neg_1)
    DECODE_LAYER_CAT_F32_32(rt, a=tensors.neg_1, b=tensors.slice_3, output=tensors.cat_1)
    DECODE_LAYER_MUL_BROADCAST_INNER_33(rt, x=tensors.cat_1, y=tensors.unsqueeze_1, output=tensors.mul_9)
    DECODE_LAYER_ADD_F32(rt, x=tensors.mul_8, y=tensors.mul_9, output=tensors.add_4)
    KV_CACHE_WRITE_F32(rt, cache=tensors.index_copy, cache_position=tensors.cache_position, src=tensors.add_4)
    KV_CACHE_WRITE_F32(rt, cache=tensors.index_copy_1, cache_position=tensors.cache_position, src=tensors.transpose_2)
    SDPA_DECODE_CACHE_F32(rt, q=tensors.add_3, k=tensors.index_copy, v=tensors.index_copy_1, cache_position=tensors.cache_position, output=tensors.scaled_dot_product_attention)
    TRANSPOSE_F32_9E77B1CEE2(rt, x=tensors.scaled_dot_product_attention, output=tensors.transpose_3)
    DECODE_LAYER_LINEAR_NOBIAS_F32_37(rt, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3)
    ADD_F32_38(rt, x=tensors.to, y=tensors.linear_3, output=tensors.add_5)
    POW_SCALAR_F32(rt, x=tensors.to_6, output=tensors.pow_4)
    DECODE_LAYER_MEAN_DIM_F32(rt, x=tensors.pow_4, output=tensors.mean_3)
    ADD_SCALAR(rt, x=tensors.mean_3, output=tensors.add_6)
    RSQRT_F32(rt, x=tensors.add_6, output=tensors.rsqrt_3)
    DECODE_LAYER_MUL_BROADCAST_LAST(rt, x=tensors.to_6, y=tensors.rsqrt_3, output=tensors.mul_10)
    MUL_LEFT_BROADCAST(rt, x=tensors.p_post_attention_layernorm_weight, y=tensors.to_7, output=tensors.mul_11)
    DECODE_LAYER_LINEAR_NOBIAS_F32_39(rt, x=tensors.mul_11, weight=tensors.p_mlp_gate_proj_weight, output=tensors.linear_4)
    SILU_F32(rt, x=tensors.linear_4, output=tensors.silu)
    DECODE_LAYER_LINEAR_NOBIAS_F32_41(rt, x=tensors.mul_11, weight=tensors.p_mlp_up_proj_weight, output=tensors.linear_5)
    MUL_F32(rt, x=tensors.silu, y=tensors.linear_5, output=tensors.mul_12)
    DECODE_LAYER_LINEAR_NOBIAS_F32_43(rt, x=tensors.mul_12, weight=tensors.p_mlp_down_proj_weight, output=tensors.linear_6)
    ADD_F32_44(rt, x=tensors.to_6, y=tensors.linear_6, output=tensors.add_7)


def _run_decode_norm_with_tensors(rt: RuntimeSession, tensors: DecodeNormTensors) -> None:
    POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    DECODE_NORM_MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    DECODE_NORM_MUL_BROADCAST_LAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    MUL_LEFT_BROADCAST(rt, x=tensors.p_weight, y=tensors.to_1, output=tensors.mul_1)


def _run_decode_lm_head_with_tensors(rt: RuntimeSession, tensors: DecodeLmHeadTensors) -> None:
    DECODE_LM_HEAD_LINEAR_NOBIAS_F32(rt, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def run_audio_encoder(rt: RuntimeSession) -> None:
    _run_audio_encoder_with_tensors(rt, model_tensors().audio_encoder)


def run_embed_tokens(rt: RuntimeSession) -> None:
    _run_embed_tokens_with_tensors(rt, model_tensors().embed_tokens)


def run_audio_inject(rt: RuntimeSession) -> None:
    _run_audio_inject_with_tensors(rt, model_tensors().audio_inject)


def run_text_layer(rt: RuntimeSession, layer_idx: int) -> None:
    _run_text_layer_with_tensors(rt, model_tensors().text_layers[layer_idx])


def run_text_norm(rt: RuntimeSession) -> None:
    _run_text_norm_with_tensors(rt, model_tensors().text_norm)


def run_lm_head(rt: RuntimeSession) -> None:
    _run_lm_head_with_tensors(rt, model_tensors().lm_head)


def run_decode_embed(rt: RuntimeSession) -> None:
    _run_decode_embed_with_tensors(rt, model_tensors().decode_embed)


def run_decode_layer(rt: RuntimeSession, layer_idx: int) -> None:
    _run_decode_layer_with_tensors(rt, model_tensors().decode_layers[layer_idx])


def run_decode_norm(rt: RuntimeSession) -> None:
    _run_decode_norm_with_tensors(rt, model_tensors().decode_norm)


def run_decode_lm_head(rt: RuntimeSession) -> None:
    _run_decode_lm_head_with_tensors(rt, model_tensors().decode_lm_head)


def run_rope_table(
    rt: RuntimeSession,
    *,
    phase: str,
    frame_name: str,
) -> None:
    tensors = model_tensors()
    if phase == "prefill":
        rope_t = tensors.prefill_rope
    elif phase == "decode":
        rope_t = tensors.decode_rope
    else:
        raise ValueError(f"unknown rope phase: {phase}")
    run_rope_table_f32(
        rt,
        start_position=rope_t.start_position,
        theta=rope_t.theta,
        cos=rope_t.cos,
        sin=rope_t.sin,
        frame_name=frame_name,
    )


def decode_step_inputs(
    *,
    cache_position: int,
    eos_token_array: np.ndarray,
    token_index_value: int,
) -> dict[LogicalTensor, np.ndarray]:
    tensors = model_tensors()
    if not tensors.decode_layers:
        raise ValueError("decode_layers must not be empty")
    return {
        tensors.decode_layers[0].cache_position: np.array([cache_position], dtype=np.int64),
        tensors.eos_token_ids: np.ascontiguousarray(eos_token_array, dtype=np.int64),
        tensors.token_index: np.array([token_index_value], dtype=np.int64),
    }


def run_decode_step(
    rt: RuntimeSession,
    *,
    step: int,
) -> int:
    tensors = model_tensors()
    if not tensors.decode_layers:
        raise ValueError("decode_layers must not be empty")
    with rt.frame(f"spike.decode.{step:04d}"):
        run_decode_embed(rt)
        for layer_idx in range(len(tensors.decode_layers)):
            run_decode_layer(rt, layer_idx)
        run_decode_norm(rt)
        run_decode_lm_head(rt)
        QWEN3_ASR_TOKEN_SELECT_GREEDY_F32(
            rt,
            logits=tensors.decode_lm_head.linear,
            eos_token_ids=tensors.eos_token_ids,
            next_token=tensors.next_token,
            done=tensors.done,
        )
        QWEN3_ASR_TOKEN_STORE_EOS_F32(
            rt,
            next_token=tensors.next_token,
            token_index=tensors.token_index,
            done=tensors.done,
            generated_tokens=tensors.generated_tokens,
            generated_length=tensors.generated_length,
            stopped=tensors.stopped,
        )
    return int(rt.read_request_state(tensors.next_token).reshape(-1)[0])
