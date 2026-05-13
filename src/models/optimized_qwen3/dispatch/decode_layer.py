"""Generated dispatch function for run_decode_layer."""

from __future__ import annotations

from models.optimized_qwen3.tensors.model import model_tensors
from models.optimized_qwen3.shaders.linear_nobias_q4_k_matvec_add_f32 import LINEAR_NOBIAS_Q4_K_MATVEC_ADD_F32
from models.optimized_qwen3.shaders.linear_nobias_q4_k_qk_matvec_f32 import LINEAR_NOBIAS_Q4_K_QK_MATVEC_F32
from models.optimized_qwen3.shaders.linear_nobias_q4_k_qkv_matvec_f32 import LINEAR_NOBIAS_Q4_K_QKV_MATVEC_F32
from models.optimized_qwen3.shaders.linear_nobias_q4_qk_q6_v_matvec_f32 import (
    LINEAR_NOBIAS_Q4_QK_Q6_V_MATVEC_F32,
)
from models.optimized_qwen3.shaders.linear_nobias_q6_k_matvec_add_f32 import LINEAR_NOBIAS_Q6_K_MATVEC_ADD_F32
from models.optimized_qwen3.shaders.rms_norm_mul_f16_f32 import RMS_NORM_MUL_F16_F32
from models.optimized_qwen3.shaders.rms_norm_rope_transpose_f16 import RMS_NORM_ROPE_TRANSPOSE_F16
from models.optimized_qwen3.shaders.sdpa_decode_cache_write_f32 import SDPA_DECODE_CACHE_WRITE_F32
from models.optimized_qwen3.shaders.swiglu_f16 import SWIGLU_F16
from models.optimized_qwen3.tensors.decode_layer import DecodeLayerTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession


def _linear_q4_or_q6_add(
    rt: RuntimeSession,
    *,
    x: LogicalTensor,
    weight: LogicalTensor,
    residual: LogicalTensor,
    output: LogicalTensor,
) -> None:
    if weight.spec.dtype == "uint16":
        LINEAR_NOBIAS_Q6_K_MATVEC_ADD_F32(rt, x=x, weight=weight, residual=residual, output=output)
        return
    LINEAR_NOBIAS_Q4_K_MATVEC_ADD_F32(rt, x=x, weight=weight, residual=residual, output=output)


def _attn_qkv(rt: RuntimeSession, tensors: DecodeLayerTensors) -> None:
    if tensors.p_attn_v_proj_weight.spec.dtype == "uint16":
        LINEAR_NOBIAS_Q4_QK_Q6_V_MATVEC_F32(
            rt,
            x=tensors.mul_1,
            q_weight=tensors.p_attn_q_proj_weight,
            k_weight=tensors.p_attn_k_proj_weight,
            v_weight=tensors.p_attn_v_proj_weight,
            q_output=tensors.linear,
            k_output=tensors.linear_1,
            v_output=tensors.linear_2,
        )
        return
    LINEAR_NOBIAS_Q4_K_QKV_MATVEC_F32(
        rt,
        x=tensors.mul_1,
        q_weight=tensors.p_attn_q_proj_weight,
        k_weight=tensors.p_attn_k_proj_weight,
        v_weight=tensors.p_attn_v_proj_weight,
        q_output=tensors.linear,
        k_output=tensors.linear_1,
        v_output=tensors.linear_2,
    )


def _run_decode_layer_with_tensors(rt: RuntimeSession, tensors: DecodeLayerTensors) -> None:
    RMS_NORM_MUL_F16_F32(
        rt,
        x=tensors.to,
        weight=tensors.p_input_layernorm_weight,
        output=tensors.mul_1,
    )
    _attn_qkv(rt, tensors)
    RMS_NORM_ROPE_TRANSPOSE_F16(
        rt,
        x=tensors.to_2,
        weight=tensors.p_attn_q_norm_weight,
        cos=tensors.position_embeddings_0,
        sin=tensors.position_embeddings_1,
        output=tensors.add_3,
    )
    RMS_NORM_ROPE_TRANSPOSE_F16(
        rt,
        x=tensors.to_4,
        weight=tensors.p_attn_k_norm_weight,
        cos=tensors.position_embeddings_0,
        sin=tensors.position_embeddings_1,
        output=tensors.add_4,
    )
    SDPA_DECODE_CACHE_WRITE_F32(
        rt,
        q=tensors.add_3,
        new_k=tensors.add_4,
        new_v=tensors.view_2,
        k_cache=tensors.index_copy,
        v_cache=tensors.index_copy_1,
        cache_position=tensors.cache_position,
        output=tensors.scaled_dot_product_attention,
    )
    LINEAR_NOBIAS_Q4_K_MATVEC_ADD_F32(
        rt,
        x=tensors.reshape,
        weight=tensors.p_attn_o_proj_weight,
        residual=tensors.to,
        output=tensors.add_5,
    )
    RMS_NORM_MUL_F16_F32(
        rt,
        x=tensors.to_6,
        weight=tensors.p_post_attention_layernorm_weight,
        output=tensors.mul_11,
    )
    LINEAR_NOBIAS_Q4_K_QK_MATVEC_F32(
        rt,
        x=tensors.mul_11,
        q_weight=tensors.p_mlp_gate_proj_weight,
        k_weight=tensors.p_mlp_up_proj_weight,
        q_output=tensors.linear_4,
        k_output=tensors.linear_5,
    )
    SWIGLU_F16(rt, gate=tensors.linear_4, up=tensors.linear_5, output=tensors.mul_12)
    _linear_q4_or_q6_add(
        rt,
        x=tensors.mul_12,
        weight=tensors.p_mlp_down_proj_weight,
        residual=tensors.to_6,
        output=tensors.add_7,
    )


def run_decode_layer(rt: RuntimeSession, layer_idx: int) -> None:
    _run_decode_layer_with_tensors(rt, model_tensors().decode_layers[layer_idx])
