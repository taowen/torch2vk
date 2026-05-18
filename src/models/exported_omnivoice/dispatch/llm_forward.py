"""Generated dispatch function for run_llm_forward."""

from __future__ import annotations

from models.exported_omnivoice.tensors.model import model_tensors
from models.exported_omnivoice.shaders.add_f32 import ADD_F32
from models.exported_omnivoice.shaders.add_f32_15 import ADD_F32_15
from models.exported_omnivoice.shaders.add_f32_18 import ADD_F32_18
from models.exported_omnivoice.shaders.cat_2_f32 import CAT_2_F32
from models.exported_omnivoice.shaders.linear_nobias_f32w_f32 import LINEAR_NOBIAS_F32W_F32
from models.exported_omnivoice.shaders.mul_broadcast import MUL_BROADCAST
from models.exported_omnivoice.shaders.mul_f32 import MUL_F32
from models.exported_omnivoice.shaders.neg_f32 import NEG_F32
from models.exported_omnivoice.shaders.rms_norm_f32w_f32 import RMS_NORM_F32W_F32
from models.exported_omnivoice.shaders.rms_norm_f32w_f32_2 import RMS_NORM_F32W_F32_2
from models.exported_omnivoice.shaders.rms_norm_f32w_f32_4 import RMS_NORM_F32W_F32_4
from models.exported_omnivoice.shaders.sdpa_masked_f32 import SDPA_MASKED_F32
from models.exported_omnivoice.shaders.silu_f32 import SILU_F32
from models.exported_omnivoice.shaders.slice_f32 import SLICE_F32
from models.exported_omnivoice.shaders.slice_f32_12 import SLICE_F32_12
from models.exported_omnivoice.shaders.slice_f32_8 import SLICE_F32_8
from models.exported_omnivoice.shaders.transpose_f32_322f87bdab import TRANSPOSE_F32_322F87BDAB
from models.exported_omnivoice.shaders.transpose_f32_b4bb3205fc import TRANSPOSE_F32_B4BB3205FC
from models.exported_omnivoice.shaders.transpose_f32_e6f353739d import TRANSPOSE_F32_E6F353739D
from models.exported_omnivoice.tensors.llm_forward import LlmForwardTensors
from torch2vk.runtime.session import RuntimeSession


def _run_llm_forward_with_tensors(rt: RuntimeSession, tensors: LlmForwardTensors) -> None:
    carry = tensors.hidden_states
    for layer_t in tensors.layers:
        RMS_NORM_F32W_F32(rt, x=carry, weight=layer_t.p_layers_0_input_layernorm_weight, output=layer_t.rms_norm)
        LINEAR_NOBIAS_F32W_F32(rt, x=layer_t.rms_norm, weight=layer_t.p_layers_0_self_attn_q_proj_weight, output=layer_t.linear)
        RMS_NORM_F32W_F32_2(rt, x=layer_t.view, weight=layer_t.p_layers_0_self_attn_q_norm_weight, output=layer_t.rms_norm_1)
        TRANSPOSE_F32_322F87BDAB(rt, x=layer_t.rms_norm_1, output=layer_t.transpose)
        LINEAR_NOBIAS_F32W_F32(rt, x=layer_t.rms_norm, weight=layer_t.p_layers_0_self_attn_k_proj_weight, output=layer_t.linear_1)
        RMS_NORM_F32W_F32_4(rt, x=layer_t.view_1, weight=layer_t.p_layers_0_self_attn_k_norm_weight, output=layer_t.rms_norm_2)
        TRANSPOSE_F32_B4BB3205FC(rt, x=layer_t.rms_norm_2, output=layer_t.transpose_1)
        LINEAR_NOBIAS_F32W_F32(rt, x=layer_t.rms_norm, weight=layer_t.p_layers_0_self_attn_v_proj_weight, output=layer_t.linear_2)
        TRANSPOSE_F32_B4BB3205FC(rt, x=layer_t.view_2, output=layer_t.transpose_2)
        MUL_BROADCAST(rt, x=layer_t.transpose, y=layer_t.unsqueeze, output=layer_t.mul)
        SLICE_F32(rt, x=layer_t.transpose, output=layer_t.slice_1)
        SLICE_F32_8(rt, x=layer_t.transpose, output=layer_t.slice_2)
        NEG_F32(rt, x=layer_t.slice_2, output=layer_t.neg)
        CAT_2_F32(rt, x0=layer_t.neg, x1=layer_t.slice_1, output=layer_t.cat)
        MUL_BROADCAST(rt, x=layer_t.cat, y=layer_t.unsqueeze_1, output=layer_t.mul_1)
        ADD_F32(rt, x=layer_t.mul, y=layer_t.mul_1, output=layer_t.add)
        MUL_BROADCAST(rt, x=layer_t.transpose_1, y=layer_t.unsqueeze, output=layer_t.mul_2)
        SLICE_F32(rt, x=layer_t.transpose_1, output=layer_t.slice_3)
        SLICE_F32_12(rt, x=layer_t.transpose_1, output=layer_t.slice_4)
        NEG_F32(rt, x=layer_t.slice_4, output=layer_t.neg_1)
        CAT_2_F32(rt, x0=layer_t.neg_1, x1=layer_t.slice_3, output=layer_t.cat_1)
        MUL_BROADCAST(rt, x=layer_t.cat_1, y=layer_t.unsqueeze_1, output=layer_t.mul_3)
        ADD_F32(rt, x=layer_t.mul_2, y=layer_t.mul_3, output=layer_t.add_1)
        SDPA_MASKED_F32(rt, q=layer_t.add, k=layer_t.add_1, v=layer_t.transpose_2, mask=tensors.attention_mask, output=layer_t.scaled_dot_product_attention)
        TRANSPOSE_F32_E6F353739D(rt, x=layer_t.scaled_dot_product_attention, output=layer_t.transpose_3)
        LINEAR_NOBIAS_F32W_F32(rt, x=layer_t.reshape, weight=layer_t.p_layers_0_self_attn_o_proj_weight, output=layer_t.linear_3)
        ADD_F32_15(rt, x=carry, y=layer_t.linear_3, output=layer_t.add_2)
        RMS_NORM_F32W_F32(rt, x=layer_t.add_2, weight=layer_t.p_layers_0_post_attention_layernorm_weight, output=layer_t.rms_norm_3)
        LINEAR_NOBIAS_F32W_F32(rt, x=layer_t.rms_norm_3, weight=layer_t.p_layers_0_mlp_gate_proj_weight, output=layer_t.linear_4)
        SILU_F32(rt, x=layer_t.linear_4, output=layer_t.silu)
        LINEAR_NOBIAS_F32W_F32(rt, x=layer_t.rms_norm_3, weight=layer_t.p_layers_0_mlp_up_proj_weight, output=layer_t.linear_5)
        MUL_F32(rt, x=layer_t.silu, y=layer_t.linear_5, output=layer_t.mul_4)
        LINEAR_NOBIAS_F32W_F32(rt, x=layer_t.mul_4, weight=layer_t.p_layers_0_mlp_down_proj_weight, output=layer_t.linear_6)
        ADD_F32_18(rt, x=layer_t.add_2, y=layer_t.linear_6, output=layer_t.add_3)
        carry = layer_t.add_3
        rt.release_layer_workspace(layer_t, layer=layer_t.add_3.layer or "", keep=(layer_t.add_3,))
    RMS_NORM_F32W_F32(rt, x=carry, weight=tensors.p_norm_weight, output=tensors.rms_norm_112)


def run_llm_forward(rt: RuntimeSession) -> None:
    tensors = model_tensors().llm_forward
    _run_llm_forward_with_tensors(rt, tensors)
