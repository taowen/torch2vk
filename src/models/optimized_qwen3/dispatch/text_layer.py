"""Generated dispatch function for run_text_layer."""

from __future__ import annotations

from models.optimized_qwen3.tensors.model import model_tensors
from models.optimized_qwen3.shaders.add_f32_33 import ADD_F32_33
from models.optimized_qwen3.shaders.add_f32_36 import ADD_F32_36
from models.optimized_qwen3.shaders.kv_cache_write_f32 import KV_CACHE_WRITE_F32
from models.optimized_qwen3.shaders.linear_nobias_q4_k_f32 import LINEAR_NOBIAS_Q4_K_F32
from models.optimized_qwen3.shaders.rms_norm_mul_4d_f16_f32 import RMS_NORM_MUL_4D_F16_F32
from models.optimized_qwen3.shaders.rms_norm_mul_f16_f32 import RMS_NORM_MUL_F16_F32
from models.optimized_qwen3.shaders.rope_transpose_f16 import ROPE_TRANSPOSE_F16
from models.optimized_qwen3.shaders.sdpa_causal_f32 import SDPA_CAUSAL_F32
from models.optimized_qwen3.shaders.swiglu_f16 import SWIGLU_F16
from models.optimized_qwen3.shaders.transpose_f32_db1599f0ff import TRANSPOSE_F32_DB1599F0FF
from models.optimized_qwen3.shaders.transpose_f32_ec88ebb72b import TRANSPOSE_F32_EC88EBB72B
from models.optimized_qwen3.tensors.text_layer import TextLayerTensors
from torch2vk.runtime.session import RuntimeSession


def _run_text_layer_with_tensors(rt: RuntimeSession, tensors: TextLayerTensors) -> None:
    RMS_NORM_MUL_F16_F32(
        rt,
        x=tensors.to,
        weight=tensors.p_input_layernorm_weight,
        output=tensors.mul_1,
    )
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.mul_1, weight=tensors.p_attn_q_proj_weight, output=tensors.linear)
    RMS_NORM_MUL_4D_F16_F32(
        rt,
        x=tensors.to_2,
        weight=tensors.p_attn_q_norm_weight,
        output=tensors.mul_3,
    )
    ROPE_TRANSPOSE_F16(
        rt,
        x=tensors.mul_3,
        cos=tensors.position_embeddings_0,
        sin=tensors.position_embeddings_1,
        output=tensors.add_3,
    )
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.mul_1, weight=tensors.p_attn_k_proj_weight, output=tensors.linear_1)
    RMS_NORM_MUL_4D_F16_F32(
        rt,
        x=tensors.to_4,
        weight=tensors.p_attn_k_norm_weight,
        output=tensors.mul_5,
    )
    ROPE_TRANSPOSE_F16(
        rt,
        x=tensors.mul_5,
        cos=tensors.position_embeddings_0,
        sin=tensors.position_embeddings_1,
        output=tensors.add_4,
    )
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.mul_1, weight=tensors.p_attn_v_proj_weight, output=tensors.linear_2)
    TRANSPOSE_F32_EC88EBB72B(rt, x=tensors.view_2, output=tensors.transpose_2)
    KV_CACHE_WRITE_F32(rt, cache=tensors.index_copy, cache_position=tensors.cache_position, src=tensors.add_4)
    KV_CACHE_WRITE_F32(rt, cache=tensors.index_copy_1, cache_position=tensors.cache_position, src=tensors.transpose_2)
    SDPA_CAUSAL_F32(rt, q=tensors.add_3, k=tensors.add_4, v=tensors.transpose_2, output=tensors.scaled_dot_product_attention)
    TRANSPOSE_F32_DB1599F0FF(rt, x=tensors.scaled_dot_product_attention, output=tensors.transpose_3)
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3)
    ADD_F32_33(rt, x=tensors.to, y=tensors.linear_3, output=tensors.add_5)
    RMS_NORM_MUL_F16_F32(
        rt,
        x=tensors.to_6,
        weight=tensors.p_post_attention_layernorm_weight,
        output=tensors.mul_11,
    )
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.mul_11, weight=tensors.p_mlp_gate_proj_weight, output=tensors.linear_4)
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.mul_11, weight=tensors.p_mlp_up_proj_weight, output=tensors.linear_5)
    SWIGLU_F16(rt, gate=tensors.linear_4, up=tensors.linear_5, output=tensors.mul_12)
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.mul_12, weight=tensors.p_mlp_down_proj_weight, output=tensors.linear_6)
    ADD_F32_36(rt, x=tensors.to_6, y=tensors.linear_6, output=tensors.add_7)


def run_text_layer(rt: RuntimeSession, layer_idx: int) -> None:
    _run_text_layer_with_tensors(rt, model_tensors().text_layers[layer_idx])
