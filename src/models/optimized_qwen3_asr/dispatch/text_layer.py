"""Generated dispatch function for run_text_layer."""

from __future__ import annotations

from models.optimized_qwen3_asr.tensors.model import model_tensors
from models.optimized_qwen3_asr.shaders.add_f32_33 import ADD_F32_33
from models.optimized_qwen3_asr.shaders.add_f32_36 import ADD_F32_36
from models.optimized_qwen3_asr.shaders.linear_nobias_q4_k_f32 import LINEAR_NOBIAS_Q4_K_F32
from models.optimized_qwen3_asr.shaders.linear_nobias_q4_k_pair_f32 import (
    LINEAR_NOBIAS_Q4_K_PAIR_F32,
)
from models.optimized_qwen3_asr.shaders.linear_nobias_q4_k_triple_f32 import (
    LINEAR_NOBIAS_Q4_K_TRIPLE_F32,
)
from models.optimized_qwen3_asr.shaders.linear_nobias_q4_k_matvec_f32 import (
    LINEAR_NOBIAS_Q4_K_MATVEC_F32,
)
from models.optimized_qwen3_asr.shaders.linear_nobias_q4_k_qk_matvec_f32 import (
    LINEAR_NOBIAS_Q4_K_QK_MATVEC_F32,
)
from models.optimized_qwen3_asr.shaders.linear_nobias_q6_k_f32 import LINEAR_NOBIAS_Q6_K_F32
from models.optimized_qwen3_asr.shaders.linear_nobias_q6_k_matvec_f32 import (
    LINEAR_NOBIAS_Q6_K_MATVEC_F32,
)
from models.optimized_qwen3_asr.shaders.rms_norm_mul_f16_f32 import RMS_NORM_MUL_F16_F32
from models.optimized_qwen3_asr.shaders.rms_norm_rope_transpose_f16 import (
    RMS_NORM_ROPE_TRANSPOSE_F16,
)
from models.optimized_qwen3_asr.shaders.sdpa_causal_cache_write_f32 import (
    SDPA_CAUSAL_CACHE_WRITE_F32,
)
from models.optimized_qwen3_asr.shaders.slice_last_token_f16 import SLICE_LAST_TOKEN_F16
from models.optimized_qwen3_asr.shaders.swiglu_f16 import SWIGLU_F16
from models.optimized_qwen3_asr.shaders.transpose_f32_db1599f0ff import TRANSPOSE_F32_DB1599F0FF
from models.optimized_qwen3_asr.tensors.text_layer import TextLayerTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession


def _linear_q4_or_q6(
    rt: RuntimeSession, *, x: LogicalTensor, weight: LogicalTensor, output: LogicalTensor
) -> None:
    if weight.spec.dtype == "uint16":
        LINEAR_NOBIAS_Q6_K_F32(rt, x=x, weight=weight, output=output)
        return
    LINEAR_NOBIAS_Q4_K_F32(rt, x=x, weight=weight, output=output)


def _linear_q4_or_q6_matvec(
    rt: RuntimeSession, *, x: LogicalTensor, weight: LogicalTensor, output: LogicalTensor
) -> None:
    if weight.spec.dtype == "uint16":
        LINEAR_NOBIAS_Q6_K_MATVEC_F32(rt, x=x, weight=weight, output=output)
        return
    LINEAR_NOBIAS_Q4_K_MATVEC_F32(rt, x=x, weight=weight, output=output)


def _attn_qkv(rt: RuntimeSession, tensors: TextLayerTensors) -> None:
    if tensors.p_attn_v_proj_weight.spec.dtype == "uint16":
        LINEAR_NOBIAS_Q4_K_PAIR_F32(
            rt,
            x=tensors.mul_1,
            weight0=tensors.p_attn_q_proj_weight,
            weight1=tensors.p_attn_k_proj_weight,
            output0=tensors.linear,
            output1=tensors.linear_1,
        )
        LINEAR_NOBIAS_Q6_K_F32(
            rt, x=tensors.mul_1, weight=tensors.p_attn_v_proj_weight, output=tensors.linear_2
        )
        return
    LINEAR_NOBIAS_Q4_K_TRIPLE_F32(
        rt,
        x=tensors.mul_1,
        weight0=tensors.p_attn_q_proj_weight,
        weight1=tensors.p_attn_k_proj_weight,
        weight2=tensors.p_attn_v_proj_weight,
        output0=tensors.linear,
        output1=tensors.linear_1,
        output2=tensors.linear_2,
    )


def _run_text_layer_with_tensors(rt: RuntimeSession, tensors: TextLayerTensors) -> None:
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
    SDPA_CAUSAL_CACHE_WRITE_F32(
        rt,
        q=tensors.add_3,
        k=tensors.add_4,
        v=tensors.view_2,
        k_cache=tensors.index_copy,
        v_cache=tensors.index_copy_1,
        cache_position=tensors.cache_position,
        output=tensors.scaled_dot_product_attention,
    )
    TRANSPOSE_F32_DB1599F0FF(rt, x=tensors.scaled_dot_product_attention, output=tensors.transpose_3)
    LINEAR_NOBIAS_Q4_K_F32(
        rt, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3
    )
    ADD_F32_33(rt, x=tensors.to, y=tensors.linear_3, output=tensors.add_5)
    RMS_NORM_MUL_F16_F32(
        rt,
        x=tensors.to_6,
        weight=tensors.p_post_attention_layernorm_weight,
        output=tensors.mul_11,
    )
    LINEAR_NOBIAS_Q4_K_PAIR_F32(
        rt,
        x=tensors.mul_11,
        weight0=tensors.p_mlp_gate_proj_weight,
        weight1=tensors.p_mlp_up_proj_weight,
        output0=tensors.linear_4,
        output1=tensors.linear_5,
    )
    SWIGLU_F16(rt, gate=tensors.linear_4, up=tensors.linear_5, output=tensors.mul_12)
    _linear_q4_or_q6(
        rt, x=tensors.mul_12, weight=tensors.p_mlp_down_proj_weight, output=tensors.linear_6
    )
    ADD_F32_36(rt, x=tensors.to_6, y=tensors.linear_6, output=tensors.add_7)


def _run_text_last_layer_tail_with_tensors(rt: RuntimeSession, tensors: TextLayerTensors) -> None:
    model = model_tensors()
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
    SDPA_CAUSAL_CACHE_WRITE_F32(
        rt,
        q=tensors.add_3,
        k=tensors.add_4,
        v=tensors.view_2,
        k_cache=tensors.index_copy,
        v_cache=tensors.index_copy_1,
        cache_position=tensors.cache_position,
        output=tensors.scaled_dot_product_attention,
    )
    TRANSPOSE_F32_DB1599F0FF(rt, x=tensors.scaled_dot_product_attention, output=tensors.transpose_3)
    LINEAR_NOBIAS_Q4_K_F32(
        rt, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3
    )
    ADD_F32_33(rt, x=tensors.to, y=tensors.linear_3, output=tensors.add_5)
    SLICE_LAST_TOKEN_F16(rt, x=tensors.add_5, output=model.prefill_last_residual)
    RMS_NORM_MUL_F16_F32(
        rt,
        x=model.prefill_last_residual,
        weight=tensors.p_post_attention_layernorm_weight,
        output=model.prefill_last_norm,
    )
    LINEAR_NOBIAS_Q4_K_QK_MATVEC_F32(
        rt,
        x=model.prefill_last_norm,
        q_weight=tensors.p_mlp_gate_proj_weight,
        k_weight=tensors.p_mlp_up_proj_weight,
        q_output=model.prefill_last_gate,
        k_output=model.prefill_last_up,
    )
    SWIGLU_F16(
        rt, gate=model.prefill_last_gate, up=model.prefill_last_up, output=model.prefill_last_gated
    )
    _linear_q4_or_q6_matvec(
        rt,
        x=model.prefill_last_gated,
        weight=tensors.p_mlp_down_proj_weight,
        output=model.prefill_last_down,
    )
    ADD_F32_36(
        rt,
        x=model.prefill_last_residual,
        y=model.prefill_last_down,
        output=model.prefill_last_output,
    )


def run_text_layer(rt: RuntimeSession, layer_idx: int) -> None:
    _run_text_layer_with_tensors(rt, model_tensors().text_layers[layer_idx])


def run_text_last_layer_tail(rt: RuntimeSession) -> None:
    _run_text_last_layer_tail_with_tensors(rt, model_tensors().text_layers[-1])
