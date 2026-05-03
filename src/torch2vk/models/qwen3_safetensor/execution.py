"""Readable Qwen3 safetensor execution records."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.logical import (
    LogicalTensor,
    MemoryPolicy,
    TensorRole,
    activation_tensor,
    input_tensor,
    output_tensor,
)
from torch2vk.shader import DispatchTarget

from .schema import qwen3_weight_tensors
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


@dataclass(frozen=True, slots=True)
class Qwen3LayerTensors:
    input: LogicalTensor
    input_norm: LogicalTensor
    q_proj: LogicalTensor
    k_proj: LogicalTensor
    v_proj: LogicalTensor
    q_heads: LogicalTensor
    k_heads: LogicalTensor
    v_rows_flat: LogicalTensor
    q_rope: LogicalTensor
    key_cache: LogicalTensor
    value_cache: LogicalTensor
    value_cache_flat: LogicalTensor
    attention_split_k: LogicalTensor
    attention_context: LogicalTensor
    attention_o_proj: LogicalTensor
    attention_residual: LogicalTensor
    post_attention_norm: LogicalTensor
    mlp_gate: LogicalTensor
    mlp_up: LogicalTensor
    mlp_gated: LogicalTensor
    output: LogicalTensor


@dataclass(frozen=True, slots=True)
class Qwen3ExecutionTensors:
    input_ids: LogicalTensor
    position_ids: LogicalTensor
    row_indices: LogicalTensor
    attention_mask: LogicalTensor
    hidden: LogicalTensor
    layers: tuple[Qwen3LayerTensors, ...]
    final_norm: LogicalTensor
    logits: LogicalTensor
    argmax_partial_values: LogicalTensor
    argmax_partial_indices: LogicalTensor
    next_token_id: LogicalTensor


def qwen3_execution_tensors(
    *,
    batch: int,
    steps: int,
    spec: Qwen3Spec,
    max_seq_len: int | None = None,
) -> Qwen3ExecutionTensors:
    cache_steps = steps if max_seq_len is None else max_seq_len
    hidden = activation_tensor(
        "decode.embedding",
        dtype="float32",
        shape=(batch, steps, spec.hidden_size),
    )
    layers: list[Qwen3LayerTensors] = []
    layer_input = hidden
    for layer_index in range(spec.num_hidden_layers):
        layer = _qwen3_layer_tensors(
            layer_index=layer_index,
            batch=batch,
            steps=steps,
            cache_steps=cache_steps,
            spec=spec,
            layer_input=layer_input,
        )
        layers.append(layer)
        layer_input = layer.output
    return Qwen3ExecutionTensors(
        input_ids=input_tensor("input.input_ids", dtype="int32", shape=(batch, steps)),
        position_ids=input_tensor("input.position_ids", dtype="int64", shape=(steps,)),
        row_indices=input_tensor("input.row_indices", dtype="int64", shape=(steps,)),
        attention_mask=activation_tensor(
            "input.attention_mask_f16",
            dtype="float16",
            shape=(batch, 1, steps, cache_steps),
            role=TensorRole.MASK,
            memory=MemoryPolicy.DEVICE_LOCAL,
        ),
        hidden=hidden,
        layers=tuple(layers),
        final_norm=activation_tensor(
            "decode.final_norm",
            dtype="float32",
            shape=(batch, steps, spec.hidden_size),
        ),
        logits=output_tensor(
            "output.logits",
            dtype="float32",
            shape=(batch, steps, spec.vocab_size),
        ),
        argmax_partial_values=activation_tensor(
            "sample.argmax.partial_values",
            dtype="float32",
            shape=(batch, _argmax_chunks(spec.vocab_size)),
            role=TensorRole.SCRATCH,
        ),
        argmax_partial_indices=activation_tensor(
            "sample.argmax.partial_indices",
            dtype="int32",
            shape=(batch, _argmax_chunks(spec.vocab_size)),
            role=TensorRole.SCRATCH,
        ),
        next_token_id=output_tensor("output.next_token_id", dtype="int32", shape=(batch,)),
    )


def record_qwen3_prefill(
    target: DispatchTarget,
    *,
    spec: Qwen3Spec,
    tensors: Qwen3ExecutionTensors,
) -> None:
    weights = {tensor.name: tensor for tensor in qwen3_weight_tensors(spec)}
    EMBEDDING_LOOKUP_BF16_F32(
        target,
        input_ids=tensors.input_ids,
        weight=weights["weights.embed_tokens"],
        output=tensors.hidden,
    )
    for layer_index, layer in enumerate(tensors.layers):
        _record_qwen3_layer(
            target,
            spec=spec,
            weights=weights,
            layer_index=layer_index,
            layer=layer,
            position_ids=tensors.position_ids,
            row_indices=tensors.row_indices,
            attention_mask=tensors.attention_mask,
        )
    RMS_NORM_F32(
        target,
        x=tensors.layers[-1].output,
        weight=weights["weights.norm"],
        output=tensors.final_norm,
    )
    LINEAR_BF16_F32(
        target,
        x=tensors.final_norm,
        weight=weights["weights.lm_head"],
        output=tensors.logits,
    )
    ARGMAX_LAST_LOGITS_STAGE1(
        target,
        logits=tensors.logits,
        partial_values=tensors.argmax_partial_values,
        partial_indices=tensors.argmax_partial_indices,
    )
    ARGMAX_LAST_LOGITS_STAGE2(
        target,
        partial_values=tensors.argmax_partial_values,
        partial_indices=tensors.argmax_partial_indices,
        output=tensors.next_token_id,
    )


def record_qwen3_minimal_prefill(
    target: DispatchTarget,
    *,
    spec: Qwen3Spec,
    tensors: Qwen3ExecutionTensors,
) -> None:
    record_qwen3_prefill(target, spec=spec, tensors=tensors)


def _record_qwen3_layer(
    target: DispatchTarget,
    *,
    spec: Qwen3Spec,
    weights: dict[str, LogicalTensor],
    layer_index: int,
    layer: Qwen3LayerTensors,
    position_ids: LogicalTensor,
    row_indices: LogicalTensor,
    attention_mask: LogicalTensor,
) -> None:
    prefix = f"weights.layer.{layer_index:02d}"
    RMS_NORM_F32(
        target,
        x=layer.input,
        weight=weights[f"{prefix}.input_layernorm"],
        output=layer.input_norm,
    )
    LINEAR_BF16_F32(
        target,
        x=layer.input_norm,
        weight=weights[f"{prefix}.self_attn.q_proj"],
        output=layer.q_proj,
    )
    LINEAR_BF16_F32(
        target,
        x=layer.input_norm,
        weight=weights[f"{prefix}.self_attn.k_proj"],
        output=layer.k_proj,
    )
    LINEAR_BF16_F32(
        target,
        x=layer.input_norm,
        weight=weights[f"{prefix}.self_attn.v_proj"],
        output=layer.v_proj,
    )
    RMS_NORM_MUL_ROPE_Q_F32(
        target,
        x=layer.q_heads,
        weight=weights[f"{prefix}.self_attn.q_norm"],
        position_ids=position_ids,
        row_indices=row_indices,
        output=layer.q_rope,
    )
    RMS_NORM_MUL_ROPE_K_F16(
        target,
        x=layer.k_heads,
        weight=weights[f"{prefix}.self_attn.k_norm"],
        position_ids=position_ids,
        row_indices=row_indices,
        output=layer.key_cache,
    )
    SET_ROWS_F16_I64_TOKEN_MAJOR(
        target,
        x=layer.v_rows_flat,
        row_indices=row_indices,
        output=layer.value_cache_flat,
    )
    FLASH_ATTN_F32_F16(
        target,
        q=layer.q_rope,
        k=layer.key_cache,
        v=layer.value_cache,
        mask=attention_mask,
        split_k_output=layer.attention_split_k,
    )
    FA_SPLIT_K_REDUCE(
        target,
        split_k_input=layer.attention_split_k,
        output=layer.attention_context,
    )
    LINEAR_BF16_F32(
        target,
        x=layer.attention_context,
        weight=weights[f"{prefix}.self_attn.o_proj"],
        output=layer.attention_o_proj,
    )
    ADD_F32(
        target,
        lhs=layer.input,
        rhs=layer.attention_o_proj,
        output=layer.attention_residual,
    )
    RMS_NORM_F32(
        target,
        x=layer.attention_residual,
        weight=weights[f"{prefix}.post_attention_layernorm"],
        output=layer.post_attention_norm,
    )
    LINEAR_BF16_F32(
        target,
        x=layer.post_attention_norm,
        weight=weights[f"{prefix}.mlp.gate_proj"],
        output=layer.mlp_gate,
    )
    LINEAR_BF16_F32(
        target,
        x=layer.post_attention_norm,
        weight=weights[f"{prefix}.mlp.up_proj"],
        output=layer.mlp_up,
    )
    SWIGLU_F32(target, gate=layer.mlp_gate, up=layer.mlp_up, output=layer.mlp_gated)
    LINEAR_BF16_F32(
        target,
        x=layer.mlp_gated,
        weight=weights[f"{prefix}.mlp.down_proj"],
        output=layer.output,
    )
    if spec.hidden_act != "silu":
        raise ValueError(f"Qwen3 safetensor port expects silu MLP, got {spec.hidden_act!r}")


def _qwen3_layer_tensors(
    *,
    layer_index: int,
    batch: int,
    steps: int,
    cache_steps: int,
    spec: Qwen3Spec,
    layer_input: LogicalTensor,
) -> Qwen3LayerTensors:
    prefix = f"decode.layer.{layer_index:02d}"
    kv_width = spec.kv_proj_out_features
    q_width = spec.q_proj_out_features
    attention_split_width = q_width + 2 * spec.num_attention_heads
    return Qwen3LayerTensors(
        input=layer_input,
        input_norm=_activation(prefix, "input_norm", batch, steps, spec.hidden_size),
        q_proj=_activation(prefix, "self_attn.q_proj", batch, steps, q_width),
        k_proj=_activation(prefix, "self_attn.k_proj", batch, steps, kv_width),
        v_proj=_activation(prefix, "self_attn.v_proj", batch, steps, kv_width),
        q_heads=_activation4(
            prefix,
            "self_attn.q_heads",
            batch,
            steps,
            spec.num_attention_heads,
            spec.head_dim,
        ),
        k_heads=_activation4(
            prefix,
            "self_attn.k_heads",
            batch,
            steps,
            spec.num_key_value_heads,
            spec.head_dim,
        ),
        v_rows_flat=_activation4(prefix, "self_attn.v_rows_flat", 1, batch, steps, kv_width),
        q_rope=_activation4(
            prefix,
            "self_attn.q_rope",
            batch,
            steps,
            spec.num_attention_heads,
            spec.head_dim,
        ),
        key_cache=_kv_cache(
            prefix,
            "key_cache",
            batch,
            cache_steps,
            spec.num_key_value_heads,
            spec.head_dim,
        ),
        value_cache=_kv_cache(
            prefix,
            "value_cache",
            batch,
            cache_steps,
            spec.num_key_value_heads,
            spec.head_dim,
        ),
        value_cache_flat=_kv_cache(prefix, "value_cache_flat", 1, batch, cache_steps, kv_width),
        attention_split_k=activation_tensor(
            f"{prefix}.self_attn.split_k",
            dtype="float32",
            shape=(batch, steps, attention_split_width * 4),
            role=TensorRole.SCRATCH,
        ),
        attention_context=_activation(prefix, "self_attn.context", batch, steps, q_width),
        attention_o_proj=_activation(prefix, "self_attn.o_proj", batch, steps, spec.hidden_size),
        attention_residual=_activation(
            prefix,
            "self_attn.residual",
            batch,
            steps,
            spec.hidden_size,
        ),
        post_attention_norm=_activation(
            prefix,
            "post_attention_norm",
            batch,
            steps,
            spec.hidden_size,
        ),
        mlp_gate=_activation(prefix, "mlp.gate", batch, steps, spec.intermediate_size),
        mlp_up=_activation(prefix, "mlp.up", batch, steps, spec.intermediate_size),
        mlp_gated=_activation(prefix, "mlp.gated", batch, steps, spec.intermediate_size),
        output=_activation(prefix, "output", batch, steps, spec.hidden_size),
    )


def _activation(prefix: str, suffix: str, batch: int, steps: int, width: int) -> LogicalTensor:
    return activation_tensor(
        f"{prefix}.{suffix}",
        dtype="float32",
        shape=(batch, steps, width),
    )


def _activation4(
    prefix: str,
    suffix: str,
    d0: int,
    d1: int,
    d2: int,
    d3: int,
) -> LogicalTensor:
    return activation_tensor(
        f"{prefix}.{suffix}",
        dtype="float32",
        shape=(d0, d1, d2, d3),
    )


def _kv_cache(prefix: str, suffix: str, d0: int, d1: int, d2: int, d3: int) -> LogicalTensor:
    return activation_tensor(
        f"{prefix}.{suffix}",
        dtype="float16",
        shape=(d0, d1, d2, d3),
        role=TensorRole.KV_CACHE,
        memory=MemoryPolicy.PERSISTENT_STATE,
    )


def _argmax_chunks(vocab_size: int) -> int:
    return (vocab_size + 1023) // 1024
