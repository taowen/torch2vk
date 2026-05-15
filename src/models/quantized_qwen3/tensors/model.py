"""Generated model-level tensor wiring for standalone Qwen3."""

from __future__ import annotations

from dataclasses import dataclass

from models.quantized_qwen3.tensors.decode_embed import DecodeEmbedTensors, create_decode_embed
from models.quantized_qwen3.tensors.decode_layer import DecodeLayerTensors, create_decode_layer
from models.quantized_qwen3.tensors.decode_norm import DecodeNormTensors, create_decode_norm
from models.quantized_qwen3.tensors.embed_tokens import EmbedTokensTensors, create_embed_tokens
from models.quantized_qwen3.tensors.lm_head import LmHeadTensors, create_lm_head
from models.quantized_qwen3.tensors.rope import RopeTableTensors, create_rope_table
from models.quantized_qwen3.tensors.text_layer import TextLayerTensors, create_text_layer
from models.quantized_qwen3.tensors.text_norm import TextNormTensors, create_text_norm
from torch2vk.runtime.logical import (
    bind_logical_tensor_names,
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    TensorSemantic,
    TensorSpec,
)


@dataclass(frozen=True, slots=True)
class QuantizedQwen3Tensors:
    input_ids: LogicalTensor
    key_caches: tuple[LogicalTensor, ...]
    value_caches: tuple[LogicalTensor, ...]
    prefill_rope: RopeTableTensors
    decode_rope: RopeTableTensors
    embed_tokens: EmbedTokensTensors
    text_layers: tuple[TextLayerTensors, ...]
    text_norm: TextNormTensors
    prefill_lm_head_input: LogicalTensor
    lm_head: LmHeadTensors
    next_token: LogicalTensor
    decode_embed: DecodeEmbedTensors
    decode_layers: tuple[DecodeLayerTensors, ...]
    decode_norm: DecodeNormTensors
    lm_head_partial_scores: LogicalTensor
    lm_head_partial_tokens: LogicalTensor
    lm_head_chunk_scores: LogicalTensor
    lm_head_chunk_tokens: LogicalTensor
    eos_token_ids: LogicalTensor
    done: LogicalTensor
    generated_tokens: LogicalTensor
    generated_length: LogicalTensor
    stopped: LogicalTensor
    token_index: LogicalTensor


_MODEL_TENSORS: QuantizedQwen3Tensors | None = None


def create_model_tensors(
    *,
    prompt_length: int,
    max_sequence_length: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    max_new_tokens: int,
    eos_token_count: int,
    vocab_size: int,
) -> QuantizedQwen3Tensors:
    input_ids = _host_input_tensor("int64", (1, prompt_length))
    key_caches = tuple(
        _request_state_tensor(
            "float16",
            (1, num_key_value_heads, max_sequence_length, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for _ in range(num_hidden_layers)
    )
    value_caches = tuple(
        _request_state_tensor(
            "float16",
            (1, num_key_value_heads, max_sequence_length, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for _ in range(num_hidden_layers)
    )
    prefill_rope = create_rope_table(
        "qwen3.prefill.rope",
        batch=1,
        sequence_length=prompt_length,
        head_dim=head_dim,
    )
    decode_rope = create_rope_table(
        "qwen3.decode.rope",
        batch=1,
        sequence_length=1,
        head_dim=head_dim,
    )
    embed_tokens = create_embed_tokens(
        "qwen3.prefill.embed",
        sequence_length=prompt_length,
        input=input_ids,
    )

    text_layers_list: list[TextLayerTensors] = []
    text_hidden = embed_tokens.embedding
    for layer_idx in range(num_hidden_layers):
        layer_tensors = create_text_layer(
            f"qwen3.prefill.layer.{layer_idx}",
            layer_idx=layer_idx,
            sequence_length=prompt_length,
            max_sequence_length=max_sequence_length,
            hidden_states=text_hidden,
            index_copy=key_caches[layer_idx],
            index_copy_1=value_caches[layer_idx],
            position_embeddings_0=prefill_rope.cos,
            position_embeddings_1=prefill_rope.sin,
            cache_position=text_layers_list[0].cache_position if layer_idx > 0 else None,
        )
        text_layers_list.append(layer_tensors)
        text_hidden = layer_tensors.add_7
    text_layers = tuple(text_layers_list)

    text_norm = create_text_norm(
        "qwen3.prefill.norm",
        sequence_length=prompt_length,
        hidden_states=text_layers[-1].add_7,
    )
    prefill_lm_head_input = _activation_tensor(
        "float16",
        (1, 1, int(text_norm.mul_1.spec.shape[-1])),
    )
    lm_head = create_lm_head("qwen3.lm_head")
    next_token = _request_output_tensor("int64", (1, 1))
    decode_embed = create_decode_embed(
        "qwen3.decode.embed",
        p_weight=embed_tokens.p_weight,
        input=next_token,
    )

    decode_layers_list: list[DecodeLayerTensors] = []
    decode_hidden = decode_embed.embedding
    for layer_idx, prefill_layer_tensors in enumerate(text_layers):
        layer_tensors = create_decode_layer(
            f"qwen3.decode.layer.{layer_idx}",
            layer_idx=layer_idx,
            max_sequence_length=max_sequence_length,
            p_input_layernorm_weight=prefill_layer_tensors.p_input_layernorm_weight,
            p_post_attention_layernorm_weight=prefill_layer_tensors.p_post_attention_layernorm_weight,
            p_attn_q_proj_weight=prefill_layer_tensors.p_attn_q_proj_weight,
            p_attn_k_proj_weight=prefill_layer_tensors.p_attn_k_proj_weight,
            p_attn_v_proj_weight=prefill_layer_tensors.p_attn_v_proj_weight,
            p_attn_o_proj_weight=prefill_layer_tensors.p_attn_o_proj_weight,
            p_attn_q_norm_weight=prefill_layer_tensors.p_attn_q_norm_weight,
            p_attn_k_norm_weight=prefill_layer_tensors.p_attn_k_norm_weight,
            p_mlp_gate_proj_weight=prefill_layer_tensors.p_mlp_gate_proj_weight,
            p_mlp_up_proj_weight=prefill_layer_tensors.p_mlp_up_proj_weight,
            p_mlp_down_proj_weight=prefill_layer_tensors.p_mlp_down_proj_weight,
            hidden_states=decode_hidden,
            index_copy=key_caches[layer_idx],
            index_copy_1=value_caches[layer_idx],
            position_embeddings_0=decode_rope.cos,
            position_embeddings_1=decode_rope.sin,
            cache_position=decode_layers_list[0].cache_position if layer_idx > 0 else None,
        )
        decode_layers_list.append(layer_tensors)
        decode_hidden = layer_tensors.add_7
    decode_layers = tuple(decode_layers_list)

    decode_norm = create_decode_norm(
        "qwen3.decode.norm",
        p_weight=text_norm.p_weight,
        hidden_states=decode_layers[-1].add_7,
    )

    eos_token_ids = _session_tensor("int64", (eos_token_count,))
    lm_head_partial_count = (vocab_size + 3) // 4
    lm_head_chunk_count = (lm_head_partial_count + 1023) // 1024
    lm_head_partial_scores = _activation_tensor("float32", (lm_head_partial_count,))
    lm_head_partial_tokens = _activation_tensor("uint32", (lm_head_partial_count,))
    lm_head_chunk_scores = _activation_tensor("float32", (lm_head_chunk_count,))
    lm_head_chunk_tokens = _activation_tensor("uint32", (lm_head_chunk_count,))
    done = _request_output_tensor("uint32", (1,))
    generated_tokens = _request_state_tensor(
        "int64",
        (1, max_new_tokens),
        semantic=TensorSemantic.TOKEN,
    )
    generated_length = _request_state_tensor(
        "uint32",
        (1,),
        semantic=TensorSemantic.TOKEN,
    )
    stopped = _request_state_tensor(
        "uint32",
        (1,),
        semantic=TensorSemantic.TOKEN,
    )
    token_index = _host_input_tensor("int64", (1,))

    global _MODEL_TENSORS
    _MODEL_TENSORS = QuantizedQwen3Tensors(
        input_ids=input_ids,
        key_caches=key_caches,
        value_caches=value_caches,
        prefill_rope=prefill_rope,
        decode_rope=decode_rope,
        embed_tokens=embed_tokens,
        text_layers=text_layers,
        text_norm=text_norm,
        prefill_lm_head_input=prefill_lm_head_input,
        lm_head=lm_head,
        next_token=next_token,
        decode_embed=decode_embed,
        decode_layers=decode_layers,
        decode_norm=decode_norm,
        lm_head_partial_scores=lm_head_partial_scores,
        lm_head_partial_tokens=lm_head_partial_tokens,
        lm_head_chunk_scores=lm_head_chunk_scores,
        lm_head_chunk_tokens=lm_head_chunk_tokens,
        eos_token_ids=eos_token_ids,
        done=done,
        generated_tokens=generated_tokens,
        generated_length=generated_length,
        stopped=stopped,
        token_index=token_index,
    )
    bind_logical_tensor_names(_MODEL_TENSORS)
    return _MODEL_TENSORS


def model_tensors() -> QuantizedQwen3Tensors:
    if _MODEL_TENSORS is None:
        raise RuntimeError("create_model_tensors must be called before generated dispatch")
    return _MODEL_TENSORS


def _host_input_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )


def _session_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.INPUT,
        memory=MemoryClass.SESSION_TENSOR,
        lifetime=TensorLifetime.MODEL,
    )


def _request_output_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.OUTPUT,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )


def _activation_tensor(dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.ACTIVATION,
        memory=MemoryClass.FRAME_WORKSPACE,
        lifetime=TensorLifetime.FRAME,
    )


def _request_state_tensor(
    dtype: str,
    shape: tuple[int, ...],
    *,
    semantic: TensorSemantic | None = None,
) -> LogicalTensor:
    return LogicalTensor(
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
        semantic=semantic,
    )
