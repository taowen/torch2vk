"""Generated dispatch function for run_text_layer."""

from __future__ import annotations

from models.quantized_qwen3.tensors.model import model_tensors
from models.quantized_qwen3.shaders.add_f32_36 import ADD_F32_36
from models.quantized_qwen3.shaders.add_f32_prefill import ADD_F32_PREFILL
from models.quantized_qwen3.shaders.fa_mask_opt_f16 import FA_MASK_OPT_F16
from models.quantized_qwen3.shaders.llama_matmul_q4_k_f32 import (
    LLAMA_MATMUL_Q4_K_F32_L,
    LLAMA_MATMUL_Q4_K_F32_M,
)
from models.quantized_qwen3.shaders.llama_matmul_q6_k_f32 import (
    LLAMA_MATMUL_Q6_K_F32_L,
    LLAMA_MATMUL_Q6_K_F32_M,
)
from models.quantized_qwen3.shaders.linear_nobias_q4_k_matvec_f32 import LINEAR_NOBIAS_Q4_K_MATVEC_F32
from models.quantized_qwen3.shaders.linear_nobias_q4_k_qk_matvec_f32 import LINEAR_NOBIAS_Q4_K_QK_MATVEC_F32
from models.quantized_qwen3.shaders.linear_nobias_q6_k_matvec_f32 import LINEAR_NOBIAS_Q6_K_MATVEC_F32
from models.quantized_qwen3.shaders.rms_norm_mul_f32 import RMS_NORM_MUL_F32
from models.quantized_qwen3.shaders.rms_norm_mul_f16_f32 import RMS_NORM_MUL_F16_F32
from models.quantized_qwen3.shaders.rms_norm_rope_token_f32 import RMS_NORM_ROPE_TOKEN_F32
from models.quantized_qwen3.shaders.rms_norm_rope_token_f32_to_f16 import (
    RMS_NORM_ROPE_TOKEN_F32_TO_F16,
)
from models.quantized_qwen3.shaders.llama_flash_attn_f32_f16_f16 import LLAMA_FLASH_ATTN_F32_F16_F16
from models.quantized_qwen3.shaders.slice_last_token_f32_to_f16 import SLICE_LAST_TOKEN_F32_TO_F16
from models.quantized_qwen3.shaders.swiglu_f32 import SWIGLU_F32
from models.quantized_qwen3.shaders.swiglu_f16 import SWIGLU_F16
from models.quantized_qwen3.shaders.token_major_kv_cache_write_f16_pair import (
    TOKEN_MAJOR_KV_CACHE_WRITE_F16_PAIR,
)
from models.quantized_qwen3.shaders.token_major_kv_cache_write_f16_triple import (
    TOKEN_MAJOR_KV_CACHE_WRITE_F16_TRIPLE,
)
from models.quantized_qwen3.shaders.token_major_value_cache_write_f32_to_f16_pair import (
    TOKEN_MAJOR_VALUE_CACHE_WRITE_F32_TO_F16_PAIR,
)
from models.quantized_qwen3.shaders.token_major_value_cache_write_f32_to_f16_triple import (
    TOKEN_MAJOR_VALUE_CACHE_WRITE_F32_TO_F16_TRIPLE,
)
from models.quantized_qwen3.tensors.text_layer import TextLayerTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession


def _linear_q4_or_q6(rt: RuntimeSession, *, x: LogicalTensor, weight: LogicalTensor, output: LogicalTensor) -> None:
    if weight.spec.dtype == "uint16":
        _llama_matmul_q6(rt, x=x, weight=weight, output=output)
        return
    _llama_matmul_q4(rt, x=x, weight=weight, output=output)


def _llama_matmul_q4(rt: RuntimeSession, *, x: LogicalTensor, weight: LogicalTensor, output: LogicalTensor) -> None:
    if _token_count(x) <= 320:
        LLAMA_MATMUL_Q4_K_F32_M(rt, x=x, weight=weight, output=output)
        return
    LLAMA_MATMUL_Q4_K_F32_L(rt, x=x, weight=weight, output=output)


def _llama_matmul_q6(rt: RuntimeSession, *, x: LogicalTensor, weight: LogicalTensor, output: LogicalTensor) -> None:
    if _token_count(x) <= 320:
        LLAMA_MATMUL_Q6_K_F32_M(rt, x=x, weight=weight, output=output)
        return
    LLAMA_MATMUL_Q6_K_F32_L(rt, x=x, weight=weight, output=output)


def _token_count(tensor: LogicalTensor) -> int:
    x0, x1, _ = tensor.spec.shape
    if not isinstance(x0, int) or not isinstance(x1, int):
        raise ValueError(f"{tensor.name} has non-concrete prefill shape {tensor.spec.shape}")
    return x0 * x1


def _linear_q4_or_q6_matvec(rt: RuntimeSession, *, x: LogicalTensor, weight: LogicalTensor, output: LogicalTensor) -> None:
    if weight.spec.dtype == "uint16":
        LINEAR_NOBIAS_Q6_K_MATVEC_F32(rt, x=x, weight=weight, output=output)
        return
    LINEAR_NOBIAS_Q4_K_MATVEC_F32(rt, x=x, weight=weight, output=output)


def _attn_qkv(rt: RuntimeSession, tensors: TextLayerTensors) -> None:
    _linear_q4_or_q6(rt, x=tensors.mul_1, weight=tensors.p_attn_q_proj_weight, output=tensors.linear)
    _linear_q4_or_q6(rt, x=tensors.mul_1, weight=tensors.p_attn_k_proj_weight, output=tensors.linear_1)
    _linear_q4_or_q6(rt, x=tensors.mul_1, weight=tensors.p_attn_v_proj_weight, output=tensors.linear_2)


def _flash_attention(
    rt: RuntimeSession,
    tensors: TextLayerTensors,
    *,
    mask: LogicalTensor,
    mask_opt: LogicalTensor,
) -> None:
    if tensors.flash_key_cache is not tensors.global_key_cache:
        TOKEN_MAJOR_KV_CACHE_WRITE_F16_TRIPLE(
            rt,
            cache_a=tensors.global_key_cache,
            cache_b=tensors.flash_key_cache,
            cache_c=tensors.decode_key_cache,
            cache_position=tensors.cache_position,
            src=tensors.add_4,
        )
    else:
        TOKEN_MAJOR_KV_CACHE_WRITE_F16_PAIR(
            rt,
            cache_a=tensors.global_key_cache,
            cache_b=tensors.decode_key_cache,
            cache_position=tensors.cache_position,
            src=tensors.add_4,
        )
    if tensors.flash_value_cache is not tensors.global_value_cache:
        TOKEN_MAJOR_VALUE_CACHE_WRITE_F32_TO_F16_TRIPLE(
            rt,
            cache_a=tensors.global_value_cache,
            cache_b=tensors.flash_value_cache,
            cache_c=tensors.decode_value_cache,
            cache_position=tensors.cache_position,
            src=tensors.view_2,
        )
    else:
        TOKEN_MAJOR_VALUE_CACHE_WRITE_F32_TO_F16_PAIR(
            rt,
            cache_a=tensors.global_value_cache,
            cache_b=tensors.decode_value_cache,
            cache_position=tensors.cache_position,
            src=tensors.view_2,
        )
    LLAMA_FLASH_ATTN_F32_F16_F16(
        rt,
        q=tensors.add_3,
        k=tensors.flash_key_cache,
        v=tensors.flash_value_cache,
        mask=mask,
        sink=tensors.add_3,
        output=tensors.transpose_3,
        mask_opt=mask_opt,
    )


def _run_text_layer_with_tensors(
    rt: RuntimeSession,
    tensors: TextLayerTensors,
    *,
    mask: LogicalTensor,
    mask_opt: LogicalTensor,
) -> None:
    RMS_NORM_MUL_F32(
        rt,
        x=tensors.to,
        weight=tensors.p_input_layernorm_weight,
        output=tensors.mul_1,
    )
    _attn_qkv(rt, tensors)
    RMS_NORM_ROPE_TOKEN_F32(
        rt,
        x=tensors.to_2,
        weight=tensors.p_attn_q_norm_weight,
        cos=tensors.position_embeddings_0,
        sin=tensors.position_embeddings_1,
        output=tensors.add_3,
    )
    RMS_NORM_ROPE_TOKEN_F32_TO_F16(
        rt,
        x=tensors.to_4,
        weight=tensors.p_attn_k_norm_weight,
        cos=tensors.position_embeddings_0,
        sin=tensors.position_embeddings_1,
        output=tensors.add_4,
    )
    _flash_attention(rt, tensors, mask=mask, mask_opt=mask_opt)
    _linear_q4_or_q6(rt, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3)
    ADD_F32_PREFILL(rt, x=tensors.to, y=tensors.linear_3, output=tensors.add_5)
    RMS_NORM_MUL_F32(
        rt,
        x=tensors.to_6,
        weight=tensors.p_post_attention_layernorm_weight,
        output=tensors.mul_11,
    )
    _linear_q4_or_q6(rt, x=tensors.mul_11, weight=tensors.p_mlp_gate_proj_weight, output=tensors.linear_4)
    _linear_q4_or_q6(rt, x=tensors.mul_11, weight=tensors.p_mlp_up_proj_weight, output=tensors.linear_5)
    SWIGLU_F32(rt, gate=tensors.linear_4, up=tensors.linear_5, output=tensors.mul_12)
    _linear_q4_or_q6(rt, x=tensors.mul_12, weight=tensors.p_mlp_down_proj_weight, output=tensors.linear_6)
    ADD_F32_PREFILL(rt, x=tensors.to_6, y=tensors.linear_6, output=tensors.add_7)


def _run_text_last_layer_tail_with_tensors(
    rt: RuntimeSession,
    tensors: TextLayerTensors,
    *,
    mask: LogicalTensor,
    mask_opt: LogicalTensor,
) -> None:
    model = model_tensors()
    RMS_NORM_MUL_F32(
        rt,
        x=tensors.to,
        weight=tensors.p_input_layernorm_weight,
        output=tensors.mul_1,
    )
    _attn_qkv(rt, tensors)
    RMS_NORM_ROPE_TOKEN_F32(rt, x=tensors.to_2, weight=tensors.p_attn_q_norm_weight, cos=tensors.position_embeddings_0, sin=tensors.position_embeddings_1, output=tensors.add_3)
    RMS_NORM_ROPE_TOKEN_F32_TO_F16(rt, x=tensors.to_4, weight=tensors.p_attn_k_norm_weight, cos=tensors.position_embeddings_0, sin=tensors.position_embeddings_1, output=tensors.add_4)
    _flash_attention(rt, tensors, mask=mask, mask_opt=mask_opt)
    _linear_q4_or_q6(rt, x=tensors.reshape, weight=tensors.p_attn_o_proj_weight, output=tensors.linear_3)
    ADD_F32_PREFILL(rt, x=tensors.to, y=tensors.linear_3, output=tensors.add_5)
    SLICE_LAST_TOKEN_F32_TO_F16(rt, x=tensors.add_5, output=model.prefill_last_residual)
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
    SWIGLU_F16(rt, gate=model.prefill_last_gate, up=model.prefill_last_up, output=model.prefill_last_gated)
    _linear_q4_or_q6_matvec(
        rt,
        x=model.prefill_last_gated,
        weight=tensors.p_mlp_down_proj_weight,
        output=model.prefill_last_down,
    )
    ADD_F32_36(rt, x=model.prefill_last_residual, y=model.prefill_last_down, output=model.prefill_last_output)


def run_prefill_full_layer(rt: RuntimeSession, layer_idx: int) -> None:
    tensors = model_tensors()
    _run_text_layer_with_tensors(
        rt,
        tensors.prefill_full_layers[layer_idx],
        mask=tensors.prefill_full_causal_mask,
        mask_opt=tensors.prefill_full_mask_opt,
    )


def run_prefill_tail_layer(rt: RuntimeSession, layer_idx: int) -> None:
    tensors = model_tensors()
    _run_text_layer_with_tensors(
        rt,
        tensors.prefill_tail_layers[layer_idx],
        mask=tensors.prefill_tail_causal_mask,
        mask_opt=tensors.prefill_tail_mask_opt,
    )


def run_prefill_tail_last_layer_tail(rt: RuntimeSession) -> None:
    tensors = model_tensors()
    _run_text_last_layer_tail_with_tensors(
        rt,
        tensors.prefill_tail_layers[-1],
        mask=tensors.prefill_tail_causal_mask,
        mask_opt=tensors.prefill_tail_mask_opt,
    )


def run_prefill_full_mask_opt(rt: RuntimeSession) -> None:
    tensors = model_tensors()
    FA_MASK_OPT_F16(
        rt,
        mask=tensors.prefill_full_causal_mask,
        output=tensors.prefill_full_mask_opt,
    )


def run_prefill_tail_mask_opt(rt: RuntimeSession) -> None:
    tensors = model_tensors()
    FA_MASK_OPT_F16(
        rt,
        mask=tensors.prefill_tail_causal_mask,
        output=tensors.prefill_tail_mask_opt,
    )
