"""Generated dispatch function for run_llm_forward."""

from __future__ import annotations

from models.quantized_omnivoice.tensors.model import model_tensors
from models.quantized_omnivoice.shaders.add_f32 import ADD_F32
from models.quantized_omnivoice.shaders.add_f32_32 import ADD_F32_32
from models.quantized_omnivoice.shaders.add_f32_35 import ADD_F32_35
from models.quantized_omnivoice.shaders.add_scalar import ADD_SCALAR
from models.quantized_omnivoice.shaders.add_scalar_16 import ADD_SCALAR_16
from models.quantized_omnivoice.shaders.add_scalar_9 import ADD_SCALAR_9
from models.quantized_omnivoice.shaders.cat_f32 import CAT_F32
from models.quantized_omnivoice.shaders.linear_nobias_q4_k_f32 import LINEAR_NOBIAS_Q4_K_F32
from models.quantized_omnivoice.shaders.mean_dim_f32 import MEAN_DIM_F32
from models.quantized_omnivoice.shaders.mean_dim_f32_15 import MEAN_DIM_F32_15
from models.quantized_omnivoice.shaders.mean_dim_f32_8 import MEAN_DIM_F32_8
from models.quantized_omnivoice.shaders.mul_broadcast_inner import MUL_BROADCAST_INNER
from models.quantized_omnivoice.shaders.mul_broadcast_inner_27 import MUL_BROADCAST_INNER_27
from models.quantized_omnivoice.shaders.mul_broadcast_inner_29 import MUL_BROADCAST_INNER_29
from models.quantized_omnivoice.shaders.mul_broadcast_last import MUL_BROADCAST_LAST
from models.quantized_omnivoice.shaders.mul_broadcast_last_11 import MUL_BROADCAST_LAST_11
from models.quantized_omnivoice.shaders.mul_broadcast_last_18 import MUL_BROADCAST_LAST_18
from models.quantized_omnivoice.shaders.mul_f32 import MUL_F32
from models.quantized_omnivoice.shaders.mul_left_broadcast_f32x_f32 import MUL_LEFT_BROADCAST_F32X_F32
from models.quantized_omnivoice.shaders.mul_left_broadcast_f32x_f32_12 import MUL_LEFT_BROADCAST_F32X_F32_12
from models.quantized_omnivoice.shaders.mul_left_broadcast_f32x_f32_19 import MUL_LEFT_BROADCAST_F32X_F32_19
from models.quantized_omnivoice.shaders.neg_f32 import NEG_F32
from models.quantized_omnivoice.shaders.pow_scalar_f32 import POW_SCALAR_F32
from models.quantized_omnivoice.shaders.pow_scalar_f32_14 import POW_SCALAR_F32_14
from models.quantized_omnivoice.shaders.pow_scalar_f32_7 import POW_SCALAR_F32_7
from models.quantized_omnivoice.shaders.rsqrt_f32 import RSQRT_F32
from models.quantized_omnivoice.shaders.rsqrt_f32_10 import RSQRT_F32_10
from models.quantized_omnivoice.shaders.rsqrt_f32_17 import RSQRT_F32_17
from models.quantized_omnivoice.shaders.sdpa_masked_f32 import SDPA_MASKED_F32
from models.quantized_omnivoice.shaders.silu_f32 import SILU_F32
from models.quantized_omnivoice.shaders.slice_f32 import SLICE_F32
from models.quantized_omnivoice.shaders.slice_f32_23 import SLICE_F32_23
from models.quantized_omnivoice.shaders.slice_f32_28 import SLICE_F32_28
from models.quantized_omnivoice.shaders.transpose_f32_322f87bdab import TRANSPOSE_F32_322F87BDAB
from models.quantized_omnivoice.shaders.transpose_f32_b4bb3205fc import TRANSPOSE_F32_B4BB3205FC
from models.quantized_omnivoice.shaders.transpose_f32_e6f353739d import TRANSPOSE_F32_E6F353739D
from models.quantized_omnivoice.tensors.llm_forward import LlmForwardTensors
from torch2vk.runtime.session import RuntimeSession


def _run_llm_forward_with_tensors(rt: RuntimeSession, tensors: LlmForwardTensors) -> None:
    for layer_t in tensors.layers:
        POW_SCALAR_F32(rt, x=layer_t.to, output=layer_t.pow_1)
        MEAN_DIM_F32(rt, x=layer_t.pow_1, output=layer_t.mean)
        ADD_SCALAR(rt, x=layer_t.mean, output=layer_t.add)
        RSQRT_F32(rt, x=layer_t.add, output=layer_t.rsqrt)
        MUL_BROADCAST_LAST(rt, x=layer_t.to, y=layer_t.rsqrt, output=layer_t.mul)
        MUL_LEFT_BROADCAST_F32X_F32(rt, x=layer_t.p_layers_0_input_layernorm_weight, y=layer_t.to_1, output=layer_t.mul_1)
        LINEAR_NOBIAS_Q4_K_F32(rt, x=layer_t.mul_1, weight=layer_t.p_layers_0_self_attn_q_proj_weight, output=layer_t.linear)
        POW_SCALAR_F32_7(rt, x=layer_t.to_2, output=layer_t.pow_2)
        MEAN_DIM_F32_8(rt, x=layer_t.pow_2, output=layer_t.mean_1)
        ADD_SCALAR_9(rt, x=layer_t.mean_1, output=layer_t.add_1)
        RSQRT_F32_10(rt, x=layer_t.add_1, output=layer_t.rsqrt_1)
        MUL_BROADCAST_LAST_11(rt, x=layer_t.to_2, y=layer_t.rsqrt_1, output=layer_t.mul_2)
        MUL_LEFT_BROADCAST_F32X_F32_12(rt, x=layer_t.p_layers_0_self_attn_q_norm_weight, y=layer_t.to_3, output=layer_t.mul_3)
        TRANSPOSE_F32_322F87BDAB(rt, x=layer_t.mul_3, output=layer_t.transpose)
        LINEAR_NOBIAS_Q4_K_F32(rt, x=layer_t.mul_1, weight=layer_t.p_layers_0_self_attn_k_proj_weight, output=layer_t.linear_1)
        POW_SCALAR_F32_14(rt, x=layer_t.to_4, output=layer_t.pow_3)
        MEAN_DIM_F32_15(rt, x=layer_t.pow_3, output=layer_t.mean_2)
        ADD_SCALAR_16(rt, x=layer_t.mean_2, output=layer_t.add_2)
        RSQRT_F32_17(rt, x=layer_t.add_2, output=layer_t.rsqrt_2)
        MUL_BROADCAST_LAST_18(rt, x=layer_t.to_4, y=layer_t.rsqrt_2, output=layer_t.mul_4)
        MUL_LEFT_BROADCAST_F32X_F32_19(rt, x=layer_t.p_layers_0_self_attn_k_norm_weight, y=layer_t.to_5, output=layer_t.mul_5)
        TRANSPOSE_F32_B4BB3205FC(rt, x=layer_t.mul_5, output=layer_t.transpose_1)
        LINEAR_NOBIAS_Q4_K_F32(rt, x=layer_t.mul_1, weight=layer_t.p_layers_0_self_attn_v_proj_weight, output=layer_t.linear_2)
        TRANSPOSE_F32_B4BB3205FC(rt, x=layer_t.view_2, output=layer_t.transpose_2)
        MUL_BROADCAST_INNER(rt, x=layer_t.transpose, y=layer_t.unsqueeze, output=layer_t.mul_6)
        SLICE_F32(rt, x=layer_t.transpose, output=layer_t.slice_1)
        SLICE_F32_23(rt, x=layer_t.transpose, output=layer_t.slice_2)
        NEG_F32(rt, x=layer_t.slice_2, output=layer_t.neg)
        CAT_F32(rt, a=layer_t.neg, b=layer_t.slice_1, output=layer_t.cat)
        MUL_BROADCAST_INNER(rt, x=layer_t.cat, y=layer_t.unsqueeze_1, output=layer_t.mul_7)
        ADD_F32(rt, x=layer_t.mul_6, y=layer_t.mul_7, output=layer_t.add_3)
        MUL_BROADCAST_INNER_27(rt, x=layer_t.transpose_1, y=layer_t.unsqueeze, output=layer_t.mul_8)
        SLICE_F32(rt, x=layer_t.transpose_1, output=layer_t.slice_3)
        SLICE_F32_28(rt, x=layer_t.transpose_1, output=layer_t.slice_4)
        NEG_F32(rt, x=layer_t.slice_4, output=layer_t.neg_1)
        CAT_F32(rt, a=layer_t.neg_1, b=layer_t.slice_3, output=layer_t.cat_1)
        MUL_BROADCAST_INNER_29(rt, x=layer_t.cat_1, y=layer_t.unsqueeze_1, output=layer_t.mul_9)
        ADD_F32(rt, x=layer_t.mul_8, y=layer_t.mul_9, output=layer_t.add_4)
        SDPA_MASKED_F32(rt, q=layer_t.add_3, k=layer_t.add_4, v=layer_t.transpose_2, mask=tensors.attention_mask, output=layer_t.scaled_dot_product_attention)
        TRANSPOSE_F32_E6F353739D(rt, x=layer_t.scaled_dot_product_attention, output=layer_t.transpose_3)
        LINEAR_NOBIAS_Q4_K_F32(rt, x=layer_t.reshape, weight=layer_t.p_layers_0_self_attn_o_proj_weight, output=layer_t.linear_3)
        ADD_F32_32(rt, x=layer_t.to, y=layer_t.linear_3, output=layer_t.add_5)
        POW_SCALAR_F32(rt, x=layer_t.to_6, output=layer_t.pow_4)
        MEAN_DIM_F32(rt, x=layer_t.pow_4, output=layer_t.mean_3)
        ADD_SCALAR(rt, x=layer_t.mean_3, output=layer_t.add_6)
        RSQRT_F32(rt, x=layer_t.add_6, output=layer_t.rsqrt_3)
        MUL_BROADCAST_LAST(rt, x=layer_t.to_6, y=layer_t.rsqrt_3, output=layer_t.mul_10)
        MUL_LEFT_BROADCAST_F32X_F32(rt, x=layer_t.p_layers_0_post_attention_layernorm_weight, y=layer_t.to_7, output=layer_t.mul_11)
        LINEAR_NOBIAS_Q4_K_F32(rt, x=layer_t.mul_11, weight=layer_t.p_layers_0_mlp_gate_proj_weight, output=layer_t.linear_4)
        SILU_F32(rt, x=layer_t.linear_4, output=layer_t.silu)
        LINEAR_NOBIAS_Q4_K_F32(rt, x=layer_t.mul_11, weight=layer_t.p_layers_0_mlp_up_proj_weight, output=layer_t.linear_5)
        MUL_F32(rt, x=layer_t.silu, y=layer_t.linear_5, output=layer_t.mul_12)
        LINEAR_NOBIAS_Q4_K_F32(rt, x=layer_t.mul_12, weight=layer_t.p_layers_0_mlp_down_proj_weight, output=layer_t.linear_6)
        ADD_F32_35(rt, x=layer_t.to_6, y=layer_t.linear_6, output=layer_t.add_7)
    POW_SCALAR_F32(rt, x=tensors.to_224, output=tensors.pow_113)
    MEAN_DIM_F32(rt, x=tensors.pow_113, output=tensors.mean_112)
    ADD_SCALAR(rt, x=tensors.mean_112, output=tensors.add_224)
    RSQRT_F32(rt, x=tensors.add_224, output=tensors.rsqrt_112)
    MUL_BROADCAST_LAST(rt, x=tensors.to_224, y=tensors.rsqrt_112, output=tensors.mul_364)
    MUL_LEFT_BROADCAST_F32X_F32(rt, x=tensors.p_norm_weight, y=tensors.to_225, output=tensors.mul_365)


def run_llm_forward(rt: RuntimeSession) -> None:
    _run_llm_forward_with_tensors(rt, model_tensors().llm_forward)
