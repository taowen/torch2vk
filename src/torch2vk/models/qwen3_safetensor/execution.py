"""Qwen3 safetensor eager shader execution."""

from __future__ import annotations

from torch2vk.logical import LogicalTensor

from .shaders.add_f32 import ADD_F32
from .shaders.argmax_last_logits_f32_stage1 import ARGMAX_LAST_LOGITS_STAGE1
from .shaders.argmax_last_logits_f32_stage2 import ARGMAX_LAST_LOGITS_STAGE2
from .shaders.embedding_lookup_bf16_f32_sequence import EMBEDDING_LOOKUP_BF16_F32
from .shaders.fa_split_k_reduce import FA_SPLIT_K_REDUCE
from .shaders.flash_attn_f32_f16 import FLASH_ATTN_F32_F16
from .shaders.linear_bf16_f32 import LINEAR_BF16_F32
from .shaders.rms_norm_f32 import RMS_NORM_F32
from .shaders.rms_norm_mul_rope_k_f16 import RMS_NORM_MUL_ROPE_K_F16
from .shaders.rms_norm_mul_rope_q_f32 import RMS_NORM_MUL_ROPE_Q_F32
from .shaders.set_rows_f16_i64_token_major import SET_ROWS_F16_I64_TOKEN_MAJOR
from .shaders.swiglu_f32 import SWIGLU_F32
from .spec import Qwen3Spec
from .tensors.decode import Qwen3DecodeTensors
from .tensors.prefill import Qwen3LayerTensors, Qwen3PrefillTensors
from .tensors.weights import Qwen3LayerWeights, Qwen3Weights


def run_qwen3_prefill(
    ctx: object,
    pytorch_model: object,
    *,
    spec: Qwen3Spec,
    tensors: Qwen3PrefillTensors,
    weights: Qwen3Weights,
) -> None:
    _run_qwen3_forward(ctx, pytorch_model, spec=spec, tensors=tensors, weights=weights)


def run_qwen3_minimal_prefill(
    ctx: object,
    pytorch_model: object,
    *,
    spec: Qwen3Spec,
    tensors: Qwen3PrefillTensors,
    weights: Qwen3Weights,
) -> None:
    run_qwen3_prefill(ctx, pytorch_model, spec=spec, tensors=tensors, weights=weights)


def run_qwen3_decode_step(
    ctx: object,
    pytorch_model: object,
    *,
    spec: Qwen3Spec,
    tensors: Qwen3DecodeTensors,
    weights: Qwen3Weights,
) -> None:
    _run_qwen3_forward(ctx, pytorch_model, spec=spec, tensors=tensors, weights=weights)


def _run_qwen3_forward(
    ctx: object,
    pytorch_model: object,
    *,
    spec: Qwen3Spec,
    tensors: Qwen3PrefillTensors | Qwen3DecodeTensors,
    weights: Qwen3Weights,
) -> None:
    EMBEDDING_LOOKUP_BF16_F32(
        ctx,
        pytorch_model,
        input_ids=tensors.input_ids,
        weight=weights.embed_tokens,
        output=tensors.hidden,
    )
    for layer_index, layer in enumerate(tensors.layers):
        _run_qwen3_layer(
            ctx,
            pytorch_model,
            spec=spec,
            weights=weights.layers[layer_index],
            layer=layer,
            position_ids=tensors.position_ids,
            row_indices=tensors.row_indices,
            rope_freq_factors_placeholder=tensors.rope_freq_factors_placeholder,
            attention_mask=tensors.attention_mask,
        )
    RMS_NORM_F32(
        ctx,
        pytorch_model,
        x=tensors.layers[-1].output,
        weight=weights.norm,
        output=tensors.final_norm,
    )
    LINEAR_BF16_F32(
        ctx,
        pytorch_model,
        x=tensors.final_norm,
        weight=weights.lm_head,
        output=tensors.logits,
    )
    ARGMAX_LAST_LOGITS_STAGE1(
        ctx,
        pytorch_model,
        logits=tensors.logits,
        partial_values=tensors.argmax_partial_values,
        partial_indices=tensors.argmax_partial_indices,
    )
    ARGMAX_LAST_LOGITS_STAGE2(
        ctx,
        pytorch_model,
        partial_values=tensors.argmax_partial_values,
        partial_indices=tensors.argmax_partial_indices,
        output=tensors.next_token_id,
    )


def _run_qwen3_layer(
    ctx: object,
    pytorch_model: object,
    *,
    spec: Qwen3Spec,
    weights: Qwen3LayerWeights,
    layer: Qwen3LayerTensors,
    position_ids: LogicalTensor,
    row_indices: LogicalTensor,
    rope_freq_factors_placeholder: LogicalTensor,
    attention_mask: LogicalTensor,
) -> None:
    RMS_NORM_F32(
        ctx,
        pytorch_model,
        x=layer.input,
        weight=weights.input_layernorm,
        output=layer.input_norm,
    )
    LINEAR_BF16_F32(
        ctx,
        pytorch_model,
        x=layer.input_norm,
        weight=weights.self_attn.q_proj,
        output=layer.q_proj,
    )
    LINEAR_BF16_F32(
        ctx,
        pytorch_model,
        x=layer.input_norm,
        weight=weights.self_attn.k_proj,
        output=layer.k_proj,
    )
    LINEAR_BF16_F32(
        ctx,
        pytorch_model,
        x=layer.input_norm,
        weight=weights.self_attn.v_proj,
        output=layer.v_proj,
    )
    RMS_NORM_MUL_ROPE_Q_F32(
        ctx,
        pytorch_model,
        x=layer.q_heads,
        weight=weights.self_attn.q_norm,
        position_ids=position_ids,
        freq_factors_placeholder=rope_freq_factors_placeholder,
        row_indices=row_indices,
        output=layer.q_rope,
    )
    RMS_NORM_MUL_ROPE_K_F16(
        ctx,
        pytorch_model,
        x=layer.k_heads,
        weight=weights.self_attn.k_norm,
        position_ids=position_ids,
        freq_factors_placeholder=rope_freq_factors_placeholder,
        row_indices=row_indices,
        output=layer.key_cache,
    )
    SET_ROWS_F16_I64_TOKEN_MAJOR(
        ctx,
        pytorch_model,
        x=layer.v_rows_flat,
        row_indices=row_indices,
        output=layer.value_cache_flat,
    )
    FLASH_ATTN_F32_F16(
        ctx,
        pytorch_model,
        q=layer.q_rope,
        k=layer.key_cache,
        v=layer.value_cache,
        mask=attention_mask,
        sinks_placeholder=layer.q_rope,
        mask_opt_placeholder=layer.q_rope,
        split_k_output=layer.attention_split_k,
    )
    FA_SPLIT_K_REDUCE(
        ctx,
        pytorch_model,
        split_k_input=layer.attention_split_k,
        sinks_placeholder=layer.attention_sinks_placeholder,
        output=layer.attention_context_heads,
    )
    LINEAR_BF16_F32(
        ctx,
        pytorch_model,
        x=layer.attention_context,
        weight=weights.self_attn.o_proj,
        output=layer.attention_o_proj,
    )
    ADD_F32(
        ctx,
        pytorch_model,
        lhs=layer.input,
        rhs=layer.attention_o_proj,
        output=layer.attention_residual,
    )
    RMS_NORM_F32(
        ctx,
        pytorch_model,
        x=layer.attention_residual,
        weight=weights.post_attention_layernorm,
        output=layer.post_attention_norm,
    )
    LINEAR_BF16_F32(
        ctx,
        pytorch_model,
        x=layer.post_attention_norm,
        weight=weights.mlp.gate_proj,
        output=layer.mlp_gate,
    )
    LINEAR_BF16_F32(
        ctx,
        pytorch_model,
        x=layer.post_attention_norm,
        weight=weights.mlp.up_proj,
        output=layer.mlp_up,
    )
    SWIGLU_F32(ctx, pytorch_model, gate=layer.mlp_gate, up=layer.mlp_up, output=layer.mlp_gated)
    LINEAR_BF16_F32(
        ctx,
        pytorch_model,
        x=layer.mlp_gated,
        weight=weights.mlp.down_proj,
        output=layer.mlp_down,
    )
    ADD_F32(
        ctx,
        pytorch_model,
        lhs=layer.attention_residual,
        rhs=layer.mlp_down,
        output=layer.output,
    )
    if spec.hidden_act != "silu":
        raise ValueError(f"Qwen3 safetensor port expects silu MLP, got {spec.hidden_act!r}")
