"""Qwen3 prefill LogicalTensor tree."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.logical import (
    ComparePolicy,
    LogicalTensor,
    MemoryPolicy,
    PyTorchProbe,
    TensorRole,
    TensorSpec,
    activation_tensor,
    input_tensor,
    output_tensor,
)
from torch2vk.models.qwen3_safetensor.spec import Qwen3Spec


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
    attention_sinks_placeholder: LogicalTensor
    attention_context_heads: LogicalTensor
    attention_context: LogicalTensor
    attention_o_proj: LogicalTensor
    attention_residual: LogicalTensor
    post_attention_norm: LogicalTensor
    mlp_gate: LogicalTensor
    mlp_up: LogicalTensor
    mlp_gated: LogicalTensor
    mlp_down: LogicalTensor
    output: LogicalTensor


@dataclass(frozen=True, slots=True)
class Qwen3PrefillTensors:
    input_ids: LogicalTensor
    position_ids: LogicalTensor
    row_indices: LogicalTensor
    rope_freq_factors_placeholder: LogicalTensor
    attention_mask: LogicalTensor
    hidden: LogicalTensor
    layers: tuple[Qwen3LayerTensors, ...]
    final_norm: LogicalTensor
    logits: LogicalTensor
    argmax_partial_values: LogicalTensor
    argmax_partial_indices: LogicalTensor
    next_token_id: LogicalTensor


def qwen3_prefill_tensors(
    *,
    batch: int,
    steps: int,
    spec: Qwen3Spec,
    max_seq_len: int | None = None,
) -> Qwen3PrefillTensors:
    cache_steps = steps if max_seq_len is None else max_seq_len
    hidden = activation_tensor(
        "decode.embedding",
        dtype="float32",
        shape=(batch, steps, spec.hidden_size),
    )
    layers: list[Qwen3LayerTensors] = []
    layer_input = hidden
    for layer_index in range(spec.num_hidden_layers):
        layer = qwen3_layer_tensors(
            layer_index=layer_index,
            batch=batch,
            steps=steps,
            cache_steps=cache_steps,
            spec=spec,
            layer_input=layer_input,
        )
        layers.append(layer)
        layer_input = layer.output
    return Qwen3PrefillTensors(
        input_ids=input_tensor("input.input_ids", dtype="int32", shape=(batch, steps)),
        position_ids=input_tensor("input.position_ids", dtype="int32", shape=(steps,)),
        row_indices=input_tensor("input.row_indices", dtype="int64", shape=(steps,)),
        rope_freq_factors_placeholder=activation_tensor(
            "input.rope_freq_factors_placeholder",
            dtype="float32",
            shape=(spec.head_dim,),
            role=TensorRole.SCRATCH,
        ),
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
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="logits",
                normalize="float32_contiguous",
            ),
            compare=ComparePolicy(kind="tensor", rtol=0.0, atol=0.5),
        ),
        argmax_partial_values=activation_tensor(
            "sample.argmax.partial_values",
            dtype="float32",
            shape=(batch, qwen3_argmax_chunks(spec.vocab_size)),
            role=TensorRole.SCRATCH,
        ),
        argmax_partial_indices=activation_tensor(
            "sample.argmax.partial_indices",
            dtype="int32",
            shape=(batch, qwen3_argmax_chunks(spec.vocab_size)),
            role=TensorRole.SCRATCH,
        ),
        next_token_id=output_tensor(
            "output.next_token_id",
            dtype="int32",
            shape=(batch,),
            pytorch_probe=PyTorchProbe(
                kind="derived",
                inputs=("output.logits",),
                transform="last_token_argmax_i32",
                normalize="int32_contiguous",
            ),
            compare=ComparePolicy(kind="token"),
        ),
    )


def qwen3_layer_tensors(
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
    attention_context_heads = _activation4(
        prefix,
        "self_attn.context",
        batch,
        steps,
        spec.num_attention_heads,
        spec.head_dim,
    )
    q_proj = _activation(prefix, "self_attn.q_proj", batch, steps, q_width)
    k_proj = _activation(prefix, "self_attn.k_proj", batch, steps, kv_width)
    v_proj = _activation(prefix, "self_attn.v_proj", batch, steps, kv_width)
    value_cache = _kv_cache(
        prefix,
        "value_cache",
        batch,
        cache_steps,
        spec.num_key_value_heads,
        spec.head_dim,
    )
    return Qwen3LayerTensors(
        input=layer_input,
        input_norm=_activation(prefix, "input_norm", batch, steps, spec.hidden_size),
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        q_heads=q_proj.view_as(
            q_proj.name,
            spec=TensorSpec(
                dtype="float32",
                shape=(batch, steps, spec.num_attention_heads, spec.head_dim),
            ),
        ),
        k_heads=k_proj.view_as(
            k_proj.name,
            spec=TensorSpec(
                dtype="float32",
                shape=(batch, steps, spec.num_key_value_heads, spec.head_dim),
            ),
        ),
        v_rows_flat=v_proj.view_as(
            v_proj.name,
            spec=TensorSpec(dtype="float32", shape=(1, batch, steps, kv_width)),
        ),
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
        value_cache=value_cache,
        value_cache_flat=value_cache.view_as(
            value_cache.name,
            spec=TensorSpec(dtype="float16", shape=(1, batch, cache_steps, kv_width)),
        ),
        attention_split_k=activation_tensor(
            f"{prefix}.self_attn.split_k",
            dtype="float32",
            shape=(batch * steps * attention_split_width * 4,),
            role=TensorRole.SCRATCH,
        ),
        attention_sinks_placeholder=_activation4(
            prefix,
            "self_attn.sinks_placeholder",
            batch,
            steps,
            spec.num_attention_heads,
            spec.head_dim,
        ),
        attention_context_heads=attention_context_heads,
        attention_context=attention_context_heads.view_as(
            f"{prefix}.self_attn.context",
            spec=TensorSpec(dtype="float32", shape=(batch, steps, q_width)),
        ),
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
        mlp_down=_activation(prefix, "mlp.down", batch, steps, spec.hidden_size),
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


def qwen3_argmax_chunks(vocab_size: int) -> int:
    return (vocab_size + 1023) // 1024
