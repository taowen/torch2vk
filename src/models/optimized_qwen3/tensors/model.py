"""Generated model-level tensor wiring for standalone Qwen3."""

from __future__ import annotations

from dataclasses import dataclass

from models.optimized_qwen3.tensors.decode_embed import DecodeEmbedTensors, create_decode_embed
from models.optimized_qwen3.tensors.decode_layer import DecodeLayerTensors, create_decode_layer
from models.optimized_qwen3.tensors.decode_norm import DecodeNormTensors, create_decode_norm
from models.optimized_qwen3.tensors.embed_tokens import EmbedTokensTensors, create_embed_tokens
from models.optimized_qwen3.tensors.lm_head import LmHeadTensors, create_lm_head
from models.optimized_qwen3.tensors.rope import RopeTableTensors, create_rope_table
from models.optimized_qwen3.tensors.text_layer import TextLayerTensors, create_text_layer
from models.optimized_qwen3.tensors.text_norm import TextNormTensors, create_text_norm
from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    TensorSemantic,
    TensorSpec,
    bind_logical_tensor_names,
)


@dataclass(frozen=True, slots=True)
class OptimizedQwen3Tensors:
    prefill_full_input_ids: LogicalTensor
    prefill_tail_input_ids: LogicalTensor
    prefill_full_causal_mask: LogicalTensor
    prefill_tail_causal_mask: LogicalTensor
    prefill_full_mask_opt: LogicalTensor
    prefill_tail_mask_opt: LogicalTensor
    prefill_full_key_caches: tuple[LogicalTensor, ...]
    prefill_full_value_caches: tuple[LogicalTensor, ...]
    prefill_key_caches: tuple[LogicalTensor, ...]
    prefill_value_caches: tuple[LogicalTensor, ...]
    decode_key_caches: tuple[LogicalTensor, ...]
    decode_value_caches: tuple[LogicalTensor, ...]
    prefill_full_rope: RopeTableTensors
    prefill_tail_rope: RopeTableTensors
    decode_rope: RopeTableTensors
    prefill_full_embed: EmbedTokensTensors
    prefill_tail_embed: EmbedTokensTensors
    prefill_full_layers: tuple[TextLayerTensors, ...]
    prefill_tail_layers: tuple[TextLayerTensors, ...]
    prefill_last_residual: LogicalTensor
    prefill_last_norm: LogicalTensor
    prefill_last_gate: LogicalTensor
    prefill_last_up: LogicalTensor
    prefill_last_gated: LogicalTensor
    prefill_last_down: LogicalTensor
    prefill_last_output: LogicalTensor
    text_norm: TextNormTensors
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


_MODEL_TENSORS: OptimizedQwen3Tensors | None = None


def create_model_tensors(
    *,
    prompt_length: int,
    prefill_chunk_length: int,
    prefill_tail_length: int,
    prefill_attention_length: int,
    max_sequence_length: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    max_new_tokens: int,
    eos_token_count: int,
    vocab_size: int,
) -> OptimizedQwen3Tensors:
    prefill_full_input_ids = _host_input_tensor("int64", (1, prefill_chunk_length))
    prefill_tail_input_ids = _host_input_tensor("int64", (1, prefill_tail_length))
    prefill_full_causal_mask = _host_input_tensor(
        "float16", (prefill_chunk_length, prefill_chunk_length)
    )
    prefill_tail_causal_mask = _host_input_tensor(
        "float16", (prefill_tail_length, prefill_attention_length)
    )
    prefill_full_mask_opt = _activation_tensor(
        "uint32",
        (_mask_opt_words(prefill_chunk_length), _mask_opt_rows(prefill_chunk_length)),
    )
    prefill_tail_mask_opt = _activation_tensor(
        "uint32",
        (_mask_opt_words(prefill_attention_length), _mask_opt_rows(prefill_tail_length)),
    )
    prefill_full_key_caches = tuple(
        _request_state_tensor(
            "float16",
            (1, prefill_chunk_length, num_key_value_heads, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for _ in range(num_hidden_layers)
    )
    prefill_full_value_caches = tuple(
        _request_state_tensor(
            "float16",
            (1, prefill_chunk_length, num_key_value_heads, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for _ in range(num_hidden_layers)
    )
    prefill_key_caches = tuple(
        _request_state_tensor(
            "float16",
            (1, prefill_attention_length, num_key_value_heads, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for _ in range(num_hidden_layers)
    )
    prefill_value_caches = tuple(
        _request_state_tensor(
            "float16",
            (1, prefill_attention_length, num_key_value_heads, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for _ in range(num_hidden_layers)
    )
    decode_key_caches = tuple(
        _request_state_tensor(
            "float16",
            (1, max_sequence_length, num_key_value_heads, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for _ in range(num_hidden_layers)
    )
    decode_value_caches = tuple(
        _request_state_tensor(
            "float16",
            (1, max_sequence_length, num_key_value_heads, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for _ in range(num_hidden_layers)
    )
    prefill_full_rope = create_rope_table(
        "qwen3.prefill.full.rope",
        batch=1,
        sequence_length=prefill_chunk_length,
        head_dim=head_dim,
    )
    prefill_tail_rope = create_rope_table(
        "qwen3.prefill.tail.rope",
        batch=1,
        sequence_length=prefill_tail_length,
        head_dim=head_dim,
    )
    decode_rope = create_rope_table(
        "qwen3.decode.rope",
        batch=1,
        sequence_length=1,
        head_dim=head_dim,
    )

    prefill_full_embed = create_embed_tokens(
        "qwen3.prefill.full.embed",
        sequence_length=prefill_chunk_length,
        input=prefill_full_input_ids,
    )

    prefill_full_layers_list: list[TextLayerTensors] = []
    text_hidden = prefill_full_embed.embedding
    for layer_idx in range(num_hidden_layers):
        layer_tensors = create_text_layer(
            f"qwen3.prefill.full.layer.{layer_idx}",
            layer_idx=layer_idx,
            num_hidden_layers=num_hidden_layers,
            sequence_length=prefill_chunk_length,
            attention_sequence_length=prefill_chunk_length,
            global_attention_sequence_length=prefill_attention_length,
            max_sequence_length=max_sequence_length,
            hidden_states=text_hidden,
            flash_key_cache=prefill_full_key_caches[layer_idx],
            flash_value_cache=prefill_full_value_caches[layer_idx],
            global_key_cache=prefill_key_caches[layer_idx],
            global_value_cache=prefill_value_caches[layer_idx],
            decode_key_cache=decode_key_caches[layer_idx],
            decode_value_cache=decode_value_caches[layer_idx],
            position_embeddings_0=prefill_full_rope.cos,
            position_embeddings_1=prefill_full_rope.sin,
            cache_position=prefill_full_layers_list[0].cache_position if layer_idx > 0 else None,
        )
        prefill_full_layers_list.append(layer_tensors)
        text_hidden = layer_tensors.add_7
    prefill_full_layers = tuple(prefill_full_layers_list)

    prefill_tail_embed = create_embed_tokens(
        "qwen3.prefill.tail.embed",
        sequence_length=prefill_tail_length,
        p_weight=prefill_full_embed.p_weight,
        input=prefill_tail_input_ids,
    )

    prefill_tail_layers_list: list[TextLayerTensors] = []
    text_hidden = prefill_tail_embed.embedding
    for layer_idx, full_layer_tensors in enumerate(prefill_full_layers):
        layer_tensors = create_text_layer(
            f"qwen3.prefill.tail.layer.{layer_idx}",
            layer_idx=layer_idx,
            num_hidden_layers=num_hidden_layers,
            sequence_length=prefill_tail_length,
            attention_sequence_length=prefill_attention_length,
            global_attention_sequence_length=prefill_attention_length,
            max_sequence_length=max_sequence_length,
            p_input_layernorm_weight=full_layer_tensors.p_input_layernorm_weight,
            p_post_attention_layernorm_weight=full_layer_tensors.p_post_attention_layernorm_weight,
            p_attn_q_proj_weight=full_layer_tensors.p_attn_q_proj_weight,
            p_attn_k_proj_weight=full_layer_tensors.p_attn_k_proj_weight,
            p_attn_v_proj_weight=full_layer_tensors.p_attn_v_proj_weight,
            p_attn_o_proj_weight=full_layer_tensors.p_attn_o_proj_weight,
            p_attn_q_norm_weight=full_layer_tensors.p_attn_q_norm_weight,
            p_attn_k_norm_weight=full_layer_tensors.p_attn_k_norm_weight,
            p_mlp_gate_proj_weight=full_layer_tensors.p_mlp_gate_proj_weight,
            p_mlp_up_proj_weight=full_layer_tensors.p_mlp_up_proj_weight,
            p_mlp_down_proj_weight=full_layer_tensors.p_mlp_down_proj_weight,
            hidden_states=text_hidden,
            flash_key_cache=prefill_key_caches[layer_idx],
            flash_value_cache=prefill_value_caches[layer_idx],
            global_key_cache=prefill_key_caches[layer_idx],
            global_value_cache=prefill_value_caches[layer_idx],
            decode_key_cache=decode_key_caches[layer_idx],
            decode_value_cache=decode_value_caches[layer_idx],
            position_embeddings_0=prefill_tail_rope.cos,
            position_embeddings_1=prefill_tail_rope.sin,
            cache_position=prefill_tail_layers_list[0].cache_position if layer_idx > 0 else None,
        )
        prefill_tail_layers_list.append(layer_tensors)
        text_hidden = layer_tensors.add_7
    prefill_tail_layers = tuple(prefill_tail_layers_list)

    prefill_last_residual = _activation_tensor("float16", (1, 1, 1024))
    prefill_last_norm = _activation_tensor("float16", (1, 1, 1024))
    prefill_last_gate = _activation_tensor("float16", (1, 1, 3072))
    prefill_last_up = _activation_tensor("float16", (1, 1, 3072))
    prefill_last_gated = _activation_tensor("float16", (1, 1, 3072))
    prefill_last_down = _activation_tensor("float16", (1, 1, 1024))
    prefill_last_output = _activation_tensor("float16", (1, 1, 1024))

    text_norm = create_text_norm(
        "qwen3.prefill.norm",
        sequence_length=1,
        hidden_states=prefill_last_output,
    )
    lm_head = create_lm_head(
        "qwen3.prefill.lm_head",
    )

    next_token = _request_output_tensor("int64", (1, 1))
    decode_embed = create_decode_embed(
        "qwen3.decode.embed",
        p_weight=prefill_full_embed.p_weight,
        input=next_token,
    )
    decode_layers_list: list[DecodeLayerTensors] = []
    decode_hidden = decode_embed.embedding
    for layer_idx, prefill_layer_tensors in enumerate(prefill_full_layers):
        layer_tensors = create_decode_layer(
            f"qwen3.decode.layer.{layer_idx}",
            layer_idx=layer_idx,
            num_hidden_layers=num_hidden_layers,
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
            index_copy=decode_key_caches[layer_idx],
            index_copy_1=decode_value_caches[layer_idx],
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
    global _MODEL_TENSORS
    _MODEL_TENSORS = OptimizedQwen3Tensors(
        prefill_full_input_ids=prefill_full_input_ids,
        prefill_tail_input_ids=prefill_tail_input_ids,
        prefill_full_causal_mask=prefill_full_causal_mask,
        prefill_tail_causal_mask=prefill_tail_causal_mask,
        prefill_full_mask_opt=prefill_full_mask_opt,
        prefill_tail_mask_opt=prefill_tail_mask_opt,
        prefill_full_key_caches=prefill_full_key_caches,
        prefill_full_value_caches=prefill_full_value_caches,
        prefill_key_caches=prefill_key_caches,
        prefill_value_caches=prefill_value_caches,
        decode_key_caches=decode_key_caches,
        decode_value_caches=decode_value_caches,
        prefill_full_rope=prefill_full_rope,
        prefill_tail_rope=prefill_tail_rope,
        decode_rope=decode_rope,
        prefill_full_embed=prefill_full_embed,
        prefill_tail_embed=prefill_tail_embed,
        prefill_full_layers=prefill_full_layers,
        prefill_tail_layers=prefill_tail_layers,
        prefill_last_residual=prefill_last_residual,
        prefill_last_norm=prefill_last_norm,
        prefill_last_gate=prefill_last_gate,
        prefill_last_up=prefill_last_up,
        prefill_last_gated=prefill_last_gated,
        prefill_last_down=prefill_last_down,
        prefill_last_output=prefill_last_output,
        text_norm=text_norm,
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
    )
    bind_logical_tensor_names(_MODEL_TENSORS)
    return _MODEL_TENSORS


def model_tensors() -> OptimizedQwen3Tensors:
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


def _mask_opt_words(attention_length: int) -> int:
    return (attention_length + 1023) // 1024


def _mask_opt_rows(sequence_length: int) -> int:
    return (sequence_length + 15) // 16


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
