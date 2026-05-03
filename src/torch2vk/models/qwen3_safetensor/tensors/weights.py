"""Qwen3 safetensor LogicalTensor weight tree."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.logical import LogicalTensor
from torch2vk.models.qwen3_safetensor.spec import Qwen3Spec
from torch2vk.schema import W


@dataclass(frozen=True, slots=True)
class Qwen3AttentionWeights:
    q_proj: LogicalTensor
    k_proj: LogicalTensor
    v_proj: LogicalTensor
    o_proj: LogicalTensor
    q_norm: LogicalTensor
    k_norm: LogicalTensor


@dataclass(frozen=True, slots=True)
class Qwen3MlpWeights:
    gate_proj: LogicalTensor
    up_proj: LogicalTensor
    down_proj: LogicalTensor


@dataclass(frozen=True, slots=True)
class Qwen3LayerWeights:
    input_layernorm: LogicalTensor
    post_attention_layernorm: LogicalTensor
    self_attn: Qwen3AttentionWeights
    mlp: Qwen3MlpWeights


@dataclass(frozen=True, slots=True)
class Qwen3Weights:
    embed_tokens: LogicalTensor
    norm: LogicalTensor
    lm_head: LogicalTensor
    layers: tuple[Qwen3LayerWeights, ...]


def qwen3_weights(spec: Qwen3Spec) -> Qwen3Weights:
    layers: list[Qwen3LayerWeights] = []
    for layer_index in range(spec.num_hidden_layers):
        checkpoint_prefix = f"model.layers.{layer_index}"
        logical_prefix = f"weights.layer.{layer_index:02d}"
        layers.append(
            Qwen3LayerWeights(
                input_layernorm=W(
                    f"{logical_prefix}.input_layernorm",
                    safetensor_key=f"{checkpoint_prefix}.input_layernorm.weight",
                    dtype="bfloat16",
                    shape=(spec.hidden_size,),
                ),
                post_attention_layernorm=W(
                    f"{logical_prefix}.post_attention_layernorm",
                    safetensor_key=f"{checkpoint_prefix}.post_attention_layernorm.weight",
                    dtype="bfloat16",
                    shape=(spec.hidden_size,),
                ),
                self_attn=Qwen3AttentionWeights(
                    q_proj=W(
                        f"{logical_prefix}.self_attn.q_proj",
                        safetensor_key=f"{checkpoint_prefix}.self_attn.q_proj.weight",
                        dtype="bfloat16",
                        shape=(spec.q_proj_out_features, spec.hidden_size),
                    ),
                    k_proj=W(
                        f"{logical_prefix}.self_attn.k_proj",
                        safetensor_key=f"{checkpoint_prefix}.self_attn.k_proj.weight",
                        dtype="bfloat16",
                        shape=(spec.kv_proj_out_features, spec.hidden_size),
                    ),
                    v_proj=W(
                        f"{logical_prefix}.self_attn.v_proj",
                        safetensor_key=f"{checkpoint_prefix}.self_attn.v_proj.weight",
                        dtype="bfloat16",
                        shape=(spec.kv_proj_out_features, spec.hidden_size),
                    ),
                    o_proj=W(
                        f"{logical_prefix}.self_attn.o_proj",
                        safetensor_key=f"{checkpoint_prefix}.self_attn.o_proj.weight",
                        dtype="bfloat16",
                        shape=(spec.hidden_size, spec.q_proj_out_features),
                    ),
                    q_norm=W(
                        f"{logical_prefix}.self_attn.q_norm",
                        safetensor_key=f"{checkpoint_prefix}.self_attn.q_norm.weight",
                        dtype="bfloat16",
                        shape=(spec.head_dim,),
                    ),
                    k_norm=W(
                        f"{logical_prefix}.self_attn.k_norm",
                        safetensor_key=f"{checkpoint_prefix}.self_attn.k_norm.weight",
                        dtype="bfloat16",
                        shape=(spec.head_dim,),
                    ),
                ),
                mlp=Qwen3MlpWeights(
                    gate_proj=W(
                        f"{logical_prefix}.mlp.gate_proj",
                        safetensor_key=f"{checkpoint_prefix}.mlp.gate_proj.weight",
                        dtype="bfloat16",
                        shape=(spec.intermediate_size, spec.hidden_size),
                    ),
                    up_proj=W(
                        f"{logical_prefix}.mlp.up_proj",
                        safetensor_key=f"{checkpoint_prefix}.mlp.up_proj.weight",
                        dtype="bfloat16",
                        shape=(spec.intermediate_size, spec.hidden_size),
                    ),
                    down_proj=W(
                        f"{logical_prefix}.mlp.down_proj",
                        safetensor_key=f"{checkpoint_prefix}.mlp.down_proj.weight",
                        dtype="bfloat16",
                        shape=(spec.hidden_size, spec.intermediate_size),
                    ),
                ),
            )
        )
    return Qwen3Weights(
        embed_tokens=W(
            "weights.embed_tokens",
            safetensor_key="model.embed_tokens.weight",
            dtype="bfloat16",
            shape=(spec.vocab_size, spec.hidden_size),
        ),
        norm=W(
            "weights.norm",
            safetensor_key="model.norm.weight",
            dtype="bfloat16",
            shape=(spec.hidden_size,),
        ),
        lm_head=W(
            "weights.lm_head",
            safetensor_key="lm_head.weight",
            dtype="bfloat16",
            shape=(spec.vocab_size, spec.hidden_size),
        ),
        layers=tuple(layers),
    )


def qwen3_weight_tensors(weights: Qwen3Weights) -> tuple[LogicalTensor, ...]:
    return (
        weights.embed_tokens,
        weights.norm,
        weights.lm_head,
        *(
            tensor
            for layer in weights.layers
            for tensor in (
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            )
        ),
    )
