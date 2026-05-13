"""Generated dispatch function for run_decode_layer."""

from __future__ import annotations

from models.quantized_qwen3.tensors.model import model_tensors
from models.quantized_qwen3.shaders.add_f32 import ADD_F32
from models.quantized_qwen3.shaders.add_f32_33 import ADD_F32_33
from models.quantized_qwen3.shaders.add_f32_36 import ADD_F32_36
from models.quantized_qwen3.shaders.add_scalar import ADD_SCALAR
from models.quantized_qwen3.shaders.add_scalar_11 import ADD_SCALAR_11
from models.quantized_qwen3.shaders.add_scalar_18 import ADD_SCALAR_18
from models.quantized_qwen3.shaders.cat_f32 import CAT_F32
from models.quantized_qwen3.shaders.decode_layer_mul_broadcast_inner import DECODE_LAYER_MUL_BROADCAST_INNER
from models.quantized_qwen3.shaders.decode_layer_mul_broadcast_last import DECODE_LAYER_MUL_BROADCAST_LAST
from models.quantized_qwen3.shaders.decode_layer_mul_broadcast_last_13 import DECODE_LAYER_MUL_BROADCAST_LAST_13
from models.quantized_qwen3.shaders.decode_layer_mul_broadcast_last_20 import DECODE_LAYER_MUL_BROADCAST_LAST_20
from models.quantized_qwen3.shaders.kv_cache_write_f16 import KV_CACHE_WRITE_F16
from models.quantized_qwen3.shaders.linear_nobias_q4_k_matvec_f32 import LINEAR_NOBIAS_Q4_K_MATVEC_F32
from models.quantized_qwen3.shaders.linear_nobias_q6_k_matvec_f32 import LINEAR_NOBIAS_Q6_K_MATVEC_F32
from models.quantized_qwen3.shaders.linear_nobias_q8_0_matvec_f32 import LINEAR_NOBIAS_Q8_0_MATVEC_F32
from models.quantized_qwen3.shaders.mean_dim_f32 import MEAN_DIM_F32
from models.quantized_qwen3.shaders.mean_dim_f32_10 import MEAN_DIM_F32_10
from models.quantized_qwen3.shaders.mean_dim_f32_17 import MEAN_DIM_F32_17
from models.quantized_qwen3.shaders.mul_f32 import MUL_F32
from models.quantized_qwen3.shaders.mul_left_broadcast_f32x_f32 import MUL_LEFT_BROADCAST_F32X_F32
from models.quantized_qwen3.shaders.mul_left_broadcast_f32x_f32_14 import MUL_LEFT_BROADCAST_F32X_F32_14
from models.quantized_qwen3.shaders.mul_left_broadcast_f32x_f32_21 import MUL_LEFT_BROADCAST_F32X_F32_21
from models.quantized_qwen3.shaders.neg_f32 import NEG_F32
from models.quantized_qwen3.shaders.pow_scalar_f32 import POW_SCALAR_F32
from models.quantized_qwen3.shaders.pow_scalar_f32_16 import POW_SCALAR_F32_16
from models.quantized_qwen3.shaders.pow_scalar_f32_9 import POW_SCALAR_F32_9
from models.quantized_qwen3.shaders.rsqrt_f32 import RSQRT_F32
from models.quantized_qwen3.shaders.rsqrt_f32_12 import RSQRT_F32_12
from models.quantized_qwen3.shaders.rsqrt_f32_19 import RSQRT_F32_19
from models.quantized_qwen3.shaders.sdpa_decode_cache_f16 import SDPA_DECODE_CACHE_F16
from models.quantized_qwen3.shaders.silu_f32 import SILU_F32
from models.quantized_qwen3.shaders.slice_f32 import SLICE_F32
from models.quantized_qwen3.shaders.slice_f32_25 import SLICE_F32_25
from models.quantized_qwen3.shaders.slice_f32_29 import SLICE_F32_29
from models.quantized_qwen3.shaders.transpose_f32_9884b7b82d import TRANSPOSE_F32_9884B7B82D
from models.quantized_qwen3.shaders.transpose_f32_9e77b1cee2 import TRANSPOSE_F32_9E77B1CEE2
from models.quantized_qwen3.shaders.transpose_f32_d509518b4f import TRANSPOSE_F32_D509518B4F
from models.quantized_qwen3.tensors.decode_layer import DecodeLayerTensors
from torch2vk.runtime.quantized_dispatch import run_quantized_linear
from torch2vk.runtime.session import RuntimeSession


def _run_decode_layer_with_tensors(rt: RuntimeSession, tensors: DecodeLayerTensors) -> None:
    POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    DECODE_LAYER_MUL_BROADCAST_LAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    MUL_LEFT_BROADCAST_F32X_F32(rt, x=tensors.p_input_layernorm_weight, y=tensors.to_1, output=tensors.mul_1)
    run_quantized_linear(rt, q4=LINEAR_NOBIAS_Q4_K_MATVEC_F32, q6=LINEAR_NOBIAS_Q6_K_MATVEC_F32, q8=LINEAR_NOBIAS_Q8_0_MATVEC_F32, x=tensors.mul_1, weight=tensors.p_attn_q_proj_weight, output=tensors.linear)
    POW_SCALAR_F32_9(rt, x=tensors.to_2, output=tensors.pow_2)
    MEAN_DIM_F32_10(rt, x=tensors.pow_2, output=tensors.mean_1)
    ADD_SCALAR_11(rt, x=tensors.mean_1, output=tensors.add_1)
    RSQRT_F32_12(rt, x=tensors.add_1, output=tensors.rsqrt_1)
    DECODE_LAYER_MUL_BROADCAST_LAST_13(rt, x=tensors.to_2, y=tensors.rsqrt_1, output=tensors.mul_2)
    MUL_LEFT_BROADCAST_F32X_F32_14(rt, x=tensors.p_attn_q_norm_weight, y=tensors.to_3, output=tensors.mul_3)
    TRANSPOSE_F32_9884B7B82D(rt, x=tensors.mul_3, output=tensors.transpose)
    run_quantized_linear(rt, q4=LINEAR_NOBIAS_Q4_K_MATVEC_F32, q6=LINEAR_NOBIAS_Q6_K_MATVEC_F32, q8=LINEAR_NOBIAS_Q8_0_MATVEC_F32, x=tensors.mul_1, weight=tensors.p_attn_k_proj_weight, output=tensors.linear_1)
    POW_SCALAR_F32_16(rt, x=tensors.to_4, output=tensors.pow_3)
    MEAN_DIM_F32_17(rt, x=tensors.pow_3, output=tensors.mean_2)
    ADD_SCALAR_18(rt, x=tensors.mean_2, output=tensors.add_2)
    RSQRT_F32_19(rt, x=tensors.add_2, output=tensors.rsqrt_2)
    DECODE_LAYER_MUL_BROADCAST_LAST_20(rt, x=tensors.to_4, y=tensors.rsqrt_2, output=tensors.mul_4)
    MUL_LEFT_BROADCAST_F32X_F32_21(rt, x=tensors.p_attn_k_norm_weight, y=tensors.to_5, output=tensors.mul_5)
    TRANSPOSE_F32_D509518B4F(rt, x=tensors.mul_5, output=tensors.transpose_1)
    run_quantized_linear(rt, q4=LINEAR_NOBIAS_Q4_K_MATVEC_F32, q6=LINEAR_NOBIAS_Q6_K_MATVEC_F32, q8=LINEAR_NOBIAS_Q8_0_MATVEC_F32, x=tensors.mul_1, weight=tensors.p_attn_v_proj_weight, output=tensors.linear_2)
    TRANSPOSE_F32_D509518B4F(rt, x=tensors.view_2, output=tensors.transpose_2)
    DECODE_LAYER_MUL_BROADCAST_INNER(rt, x=tensors.transpose, y=tensors.unsqueeze, output=tensors.mul_6)
    SLICE_F32(rt, x=tensors.transpose, output=tensors.slice_1)
    SLICE_F32_25(rt, x=tensors.transpose, output=tensors.slice_2)
    NEG_F32(rt, x=tensors.slice_2, output=tensors.neg)
    CAT_F32(rt, a=tensors.neg, b=tensors.slice_1, output=tensors.cat)
    DECODE_LAYER_MUL_BROADCAST_INNER(rt, x=tensors.cat, y=tensors.unsqueeze_1, output=tensors.mul_7)
    ADD_F32(rt, x=tensors.mul_6, y=tensors.mul_7, output=tensors.add_3)
    DECODE_LAYER_MUL_BROADCAST_INNER(rt, x=tensors.transpose_1, y=tensors.unsqueeze, output=tensors.mul_8)
    SLICE_F32(rt, x=tensors.transpose_1, output=tensors.slice_3)
    SLICE_F32_29(rt, x=tensors.transpose_1, output=tensors.slice_4)
    NEG_F32(rt, x=tensors.slice_4, output=tensors.neg_1)
    CAT_F32(rt, a=tensors.neg_1, b=tensors.slice_3, output=tensors.cat_1)
    DECODE_LAYER_MUL_BROADCAST_INNER(rt, x=tensors.cat_1, y=tensors.unsqueeze_1, output=tensors.mul_9)
    ADD_F32(rt, x=tensors.mul_8, y=tensors.mul_9, output=tensors.add_4)
    KV_CACHE_WRITE_F16(rt, cache=tensors.index_copy, cache_position=tensors.cache_position, src=tensors.add_4)
    KV_CACHE_WRITE_F16(rt, cache=tensors.index_copy_1, cache_position=tensors.cache_position, src=tensors.transpose_2)
    SDPA_DECODE_CACHE_F16(rt, q=tensors.add_3, k=tensors.index_copy, v=tensors.index_copy_1, cache_position=tensors.cache_position, output=tensors.scaled_dot_product_attention)
    TRANSPOSE_F32_9E77B1CEE2(rt, x=tensors.scaled_dot_product_attention, output=tensors.transpose_3)
    run_quantized_linear(rt, q4=LINEAR_NOBIAS_Q4_K_MATVEC_F32, q6=LINEAR_NOBIAS_Q6_K_MATVEC_F32, q8=LINEAR_NOBIAS_Q8_0_MATVEC_F32, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3)
    ADD_F32_33(rt, x=tensors.to, y=tensors.linear_3, output=tensors.add_5)
    POW_SCALAR_F32(rt, x=tensors.to_6, output=tensors.pow_4)
    MEAN_DIM_F32(rt, x=tensors.pow_4, output=tensors.mean_3)
    ADD_SCALAR(rt, x=tensors.mean_3, output=tensors.add_6)
    RSQRT_F32(rt, x=tensors.add_6, output=tensors.rsqrt_3)
    DECODE_LAYER_MUL_BROADCAST_LAST(rt, x=tensors.to_6, y=tensors.rsqrt_3, output=tensors.mul_10)
    MUL_LEFT_BROADCAST_F32X_F32(rt, x=tensors.p_post_attention_layernorm_weight, y=tensors.to_7, output=tensors.mul_11)
    run_quantized_linear(rt, q4=LINEAR_NOBIAS_Q4_K_MATVEC_F32, q6=LINEAR_NOBIAS_Q6_K_MATVEC_F32, q8=LINEAR_NOBIAS_Q8_0_MATVEC_F32, x=tensors.mul_11, weight=tensors.p_mlp_gate_proj_weight, output=tensors.linear_4)
    SILU_F32(rt, x=tensors.linear_4, output=tensors.silu)
    run_quantized_linear(rt, q4=LINEAR_NOBIAS_Q4_K_MATVEC_F32, q6=LINEAR_NOBIAS_Q6_K_MATVEC_F32, q8=LINEAR_NOBIAS_Q8_0_MATVEC_F32, x=tensors.mul_11, weight=tensors.p_mlp_up_proj_weight, output=tensors.linear_5)
    MUL_F32(rt, x=tensors.silu, y=tensors.linear_5, output=tensors.mul_12)
    run_quantized_linear(rt, q4=LINEAR_NOBIAS_Q4_K_MATVEC_F32, q6=LINEAR_NOBIAS_Q6_K_MATVEC_F32, q8=LINEAR_NOBIAS_Q8_0_MATVEC_F32, x=tensors.mul_12, weight=tensors.p_mlp_down_proj_weight, output=tensors.linear_6)
    ADD_F32_36(rt, x=tensors.to_6, y=tensors.linear_6, output=tensors.add_7)


def run_decode_layer(rt: RuntimeSession, layer_idx: int) -> None:
    _run_decode_layer_with_tensors(rt, model_tensors().decode_layers[layer_idx])
