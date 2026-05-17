"""Generated dispatch function for run_decode_layer."""

from __future__ import annotations

from models.quantized_qwen3_asr.tensors.model import model_tensors
from models.quantized_qwen3_asr.shaders.add_f32_36 import ADD_F32_36
from models.quantized_qwen3_asr.shaders.add_f32_39 import ADD_F32_39
from models.quantized_qwen3_asr.shaders.add_scalar import ADD_SCALAR
from models.quantized_qwen3_asr.shaders.add_scalar_16 import ADD_SCALAR_16
from models.quantized_qwen3_asr.shaders.add_scalar_9 import ADD_SCALAR_9
from models.quantized_qwen3_asr.shaders.cat_2_f32 import CAT_2_F32
from models.quantized_qwen3_asr.shaders.decode_layer_add_f32 import DECODE_LAYER_ADD_F32
from models.quantized_qwen3_asr.shaders.decode_layer_mul_broadcast import DECODE_LAYER_MUL_BROADCAST
from models.quantized_qwen3_asr.shaders.decode_layer_mul_broadcast_11 import DECODE_LAYER_MUL_BROADCAST_11
from models.quantized_qwen3_asr.shaders.decode_layer_mul_broadcast_18 import DECODE_LAYER_MUL_BROADCAST_18
from models.quantized_qwen3_asr.shaders.decode_layer_mul_broadcast_23 import DECODE_LAYER_MUL_BROADCAST_23
from models.quantized_qwen3_asr.shaders.decode_layer_mul_broadcast_28 import DECODE_LAYER_MUL_BROADCAST_28
from models.quantized_qwen3_asr.shaders.decode_layer_mul_broadcast_30 import DECODE_LAYER_MUL_BROADCAST_30
from models.quantized_qwen3_asr.shaders.decode_layer_mul_broadcast_32 import DECODE_LAYER_MUL_BROADCAST_32
from models.quantized_qwen3_asr.shaders.kv_cache_write_decode_f16 import KV_CACHE_WRITE_DECODE_F16
from models.quantized_qwen3_asr.shaders.linear_nobias_q4_k_matvec_f32 import LINEAR_NOBIAS_Q4_K_MATVEC_F32
from models.quantized_qwen3_asr.shaders.linear_nobias_q6_k_matvec_f32 import LINEAR_NOBIAS_Q6_K_MATVEC_F32
from models.quantized_qwen3_asr.shaders.linear_nobias_q8_0_matvec_f32 import LINEAR_NOBIAS_Q8_0_MATVEC_F32
from models.quantized_qwen3_asr.shaders.mean_dim_f32 import MEAN_DIM_F32
from models.quantized_qwen3_asr.shaders.mean_dim_f32_15 import MEAN_DIM_F32_15
from models.quantized_qwen3_asr.shaders.mean_dim_f32_8 import MEAN_DIM_F32_8
from models.quantized_qwen3_asr.shaders.mul_f32 import MUL_F32
from models.quantized_qwen3_asr.shaders.mul_left_broadcast_f32x_f32 import MUL_LEFT_BROADCAST_F32X_F32
from models.quantized_qwen3_asr.shaders.mul_left_broadcast_f32x_f32_12 import MUL_LEFT_BROADCAST_F32X_F32_12
from models.quantized_qwen3_asr.shaders.mul_left_broadcast_f32x_f32_19 import MUL_LEFT_BROADCAST_F32X_F32_19
from models.quantized_qwen3_asr.shaders.neg_f32 import NEG_F32
from models.quantized_qwen3_asr.shaders.pow_scalar_f32 import POW_SCALAR_F32
from models.quantized_qwen3_asr.shaders.pow_scalar_f32_14 import POW_SCALAR_F32_14
from models.quantized_qwen3_asr.shaders.pow_scalar_f32_7 import POW_SCALAR_F32_7
from models.quantized_qwen3_asr.shaders.rsqrt_f32 import RSQRT_F32
from models.quantized_qwen3_asr.shaders.rsqrt_f32_10 import RSQRT_F32_10
from models.quantized_qwen3_asr.shaders.rsqrt_f32_17 import RSQRT_F32_17
from models.quantized_qwen3_asr.shaders.sdpa_decode_cache_f16 import SDPA_DECODE_CACHE_F16
from models.quantized_qwen3_asr.shaders.silu_f32 import SILU_F32
from models.quantized_qwen3_asr.shaders.slice_f32 import SLICE_F32
from models.quantized_qwen3_asr.shaders.slice_f32_25 import SLICE_F32_25
from models.quantized_qwen3_asr.shaders.slice_f32_31 import SLICE_F32_31
from models.quantized_qwen3_asr.shaders.transpose_f32_9884b7b82d import TRANSPOSE_F32_9884B7B82D
from models.quantized_qwen3_asr.shaders.transpose_f32_9e77b1cee2 import TRANSPOSE_F32_9E77B1CEE2
from models.quantized_qwen3_asr.shaders.transpose_f32_d509518b4f import TRANSPOSE_F32_D509518B4F
from models.quantized_qwen3_asr.tensors.decode_layer import DecodeLayerTensors
from torch2vk.runtime.quantized_dispatch import run_quantized_linear
from torch2vk.runtime.session import RuntimeSession


def _run_decode_layer_with_tensors(rt: RuntimeSession, tensors: DecodeLayerTensors, *, cache_position: int) -> None:
    POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    DECODE_LAYER_MUL_BROADCAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    MUL_LEFT_BROADCAST_F32X_F32(rt, x=tensors.p_input_layernorm_weight, y=tensors.to_1, output=tensors.mul_1)
    LINEAR_NOBIAS_Q4_K_MATVEC_F32(rt, x=tensors.mul_1, weight=tensors.p_attn_q_proj_weight, output=tensors.linear)
    POW_SCALAR_F32_7(rt, x=tensors.to_2, output=tensors.pow_2)
    MEAN_DIM_F32_8(rt, x=tensors.pow_2, output=tensors.mean_1)
    ADD_SCALAR_9(rt, x=tensors.mean_1, output=tensors.add_1)
    RSQRT_F32_10(rt, x=tensors.add_1, output=tensors.rsqrt_1)
    DECODE_LAYER_MUL_BROADCAST_11(rt, x=tensors.to_2, y=tensors.rsqrt_1, output=tensors.mul_2)
    MUL_LEFT_BROADCAST_F32X_F32_12(rt, x=tensors.p_attn_q_norm_weight, y=tensors.to_3, output=tensors.mul_3)
    TRANSPOSE_F32_9884B7B82D(rt, x=tensors.mul_3, output=tensors.transpose)
    LINEAR_NOBIAS_Q4_K_MATVEC_F32(rt, x=tensors.mul_1, weight=tensors.p_attn_k_proj_weight, output=tensors.linear_1)
    POW_SCALAR_F32_14(rt, x=tensors.to_4, output=tensors.pow_3)
    MEAN_DIM_F32_15(rt, x=tensors.pow_3, output=tensors.mean_2)
    ADD_SCALAR_16(rt, x=tensors.mean_2, output=tensors.add_2)
    RSQRT_F32_17(rt, x=tensors.add_2, output=tensors.rsqrt_2)
    DECODE_LAYER_MUL_BROADCAST_18(rt, x=tensors.to_4, y=tensors.rsqrt_2, output=tensors.mul_4)
    MUL_LEFT_BROADCAST_F32X_F32_19(rt, x=tensors.p_attn_k_norm_weight, y=tensors.to_5, output=tensors.mul_5)
    TRANSPOSE_F32_D509518B4F(rt, x=tensors.mul_5, output=tensors.transpose_1)
    run_quantized_linear(rt, q4=LINEAR_NOBIAS_Q4_K_MATVEC_F32, q6=LINEAR_NOBIAS_Q6_K_MATVEC_F32, q8=LINEAR_NOBIAS_Q8_0_MATVEC_F32, x=tensors.mul_1, weight=tensors.p_attn_v_proj_weight, output=tensors.linear_2)
    TRANSPOSE_F32_D509518B4F(rt, x=tensors.view_2, output=tensors.transpose_2)
    DECODE_LAYER_MUL_BROADCAST_23(rt, x=tensors.transpose, y=tensors.unsqueeze, output=tensors.mul_6)
    SLICE_F32(rt, x=tensors.transpose, output=tensors.slice_1)
    SLICE_F32_25(rt, x=tensors.transpose, output=tensors.slice_2)
    NEG_F32(rt, x=tensors.slice_2, output=tensors.neg)
    CAT_2_F32(rt, x0=tensors.neg, x1=tensors.slice_1, output=tensors.cat)
    DECODE_LAYER_MUL_BROADCAST_28(rt, x=tensors.cat, y=tensors.unsqueeze_1, output=tensors.mul_7)
    DECODE_LAYER_ADD_F32(rt, x=tensors.mul_6, y=tensors.mul_7, output=tensors.add_3)
    DECODE_LAYER_MUL_BROADCAST_30(rt, x=tensors.transpose_1, y=tensors.unsqueeze, output=tensors.mul_8)
    SLICE_F32(rt, x=tensors.transpose_1, output=tensors.slice_3)
    SLICE_F32_31(rt, x=tensors.transpose_1, output=tensors.slice_4)
    NEG_F32(rt, x=tensors.slice_4, output=tensors.neg_1)
    CAT_2_F32(rt, x0=tensors.neg_1, x1=tensors.slice_3, output=tensors.cat_1)
    DECODE_LAYER_MUL_BROADCAST_32(rt, x=tensors.cat_1, y=tensors.unsqueeze_1, output=tensors.mul_9)
    DECODE_LAYER_ADD_F32(rt, x=tensors.mul_8, y=tensors.mul_9, output=tensors.add_4)
    KV_CACHE_WRITE_DECODE_F16(rt, cache=tensors.index_copy, src=tensors.add_4, cache_position=cache_position)
    KV_CACHE_WRITE_DECODE_F16(rt, cache=tensors.index_copy_1, src=tensors.transpose_2, cache_position=cache_position)
    SDPA_DECODE_CACHE_F16(rt, q=tensors.add_3, k=tensors.index_copy, v=tensors.index_copy_1, output=tensors.scaled_dot_product_attention, cache_position=cache_position)
    TRANSPOSE_F32_9E77B1CEE2(rt, x=tensors.scaled_dot_product_attention, output=tensors.transpose_3)
    LINEAR_NOBIAS_Q4_K_MATVEC_F32(rt, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3)
    ADD_F32_36(rt, x=tensors.to, y=tensors.linear_3, output=tensors.add_5)
    POW_SCALAR_F32(rt, x=tensors.to_6, output=tensors.pow_4)
    MEAN_DIM_F32(rt, x=tensors.pow_4, output=tensors.mean_3)
    ADD_SCALAR(rt, x=tensors.mean_3, output=tensors.add_6)
    RSQRT_F32(rt, x=tensors.add_6, output=tensors.rsqrt_3)
    DECODE_LAYER_MUL_BROADCAST(rt, x=tensors.to_6, y=tensors.rsqrt_3, output=tensors.mul_10)
    MUL_LEFT_BROADCAST_F32X_F32(rt, x=tensors.p_post_attention_layernorm_weight, y=tensors.to_7, output=tensors.mul_11)
    LINEAR_NOBIAS_Q4_K_MATVEC_F32(rt, x=tensors.mul_11, weight=tensors.p_mlp_gate_proj_weight, output=tensors.linear_4)
    SILU_F32(rt, x=tensors.linear_4, output=tensors.silu)
    LINEAR_NOBIAS_Q4_K_MATVEC_F32(rt, x=tensors.mul_11, weight=tensors.p_mlp_up_proj_weight, output=tensors.linear_5)
    MUL_F32(rt, x=tensors.silu, y=tensors.linear_5, output=tensors.mul_12)
    run_quantized_linear(rt, q4=LINEAR_NOBIAS_Q4_K_MATVEC_F32, q6=LINEAR_NOBIAS_Q6_K_MATVEC_F32, q8=LINEAR_NOBIAS_Q8_0_MATVEC_F32, x=tensors.mul_12, weight=tensors.p_mlp_down_proj_weight, output=tensors.linear_6)
    ADD_F32_39(rt, x=tensors.to_6, y=tensors.linear_6, output=tensors.add_7)


def run_decode_layer(rt: RuntimeSession, layer_idx: int, *, cache_position: int) -> None:
    tensors = model_tensors().decode_layers[layer_idx]
    _run_decode_layer_with_tensors(rt, tensors, cache_position=cache_position)
    rt.release_layer_workspace(
        tensors,
        layer=tensors.add_7.layer or "",
        keep=(
            tensors.add_7,
            tensors.index_copy,
            tensors.index_copy_1,
        ),
    )
