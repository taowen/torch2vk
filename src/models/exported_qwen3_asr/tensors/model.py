"""Generated model-level tensor wiring."""

from __future__ import annotations

from dataclasses import dataclass

from models.exported_qwen3_asr.tensors.audio_encoder import (
    AUDIO_ENCODER_OUTPUT,
    AudioEncoderTensors,
    create_audio_encoder,
)
from models.exported_qwen3_asr.tensors.audio_inject import (
    AudioInjectTensors,
    create_audio_inject,
)
from models.exported_qwen3_asr.tensors.decode_embed import (
    DecodeEmbedTensors,
    create_decode_embed,
)
from models.exported_qwen3_asr.tensors.decode_layer import (
    DecodeLayerTensors,
    create_decode_layer,
)
from models.exported_qwen3_asr.tensors.decode_lm_head import (
    DECODE_LM_HEAD_OUTPUT,
    DecodeLmHeadTensors,
    create_decode_lm_head,
)
from models.exported_qwen3_asr.tensors.decode_norm import (
    DecodeNormTensors,
    create_decode_norm,
)
from models.exported_qwen3_asr.tensors.embed_tokens import (
    EmbedTokensTensors,
    create_embed_tokens,
)
from models.exported_qwen3_asr.tensors.lm_head import (
    LM_HEAD_OUTPUT,
    LmHeadTensors,
    create_lm_head,
)
from models.exported_qwen3_asr.tensors.rope import RopeTableTensors, create_rope_table
from models.exported_qwen3_asr.tensors.text_layer import TextLayerTensors, create_text_layer
from models.exported_qwen3_asr.tensors.text_norm import TextNormTensors, create_text_norm
from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    TensorSemantic,
    TensorSpec,
)


@dataclass(frozen=True, slots=True)
class ExportedQwen3AsrTensors:
    input_ids_t: LogicalTensor
    attention_mask_t: LogicalTensor
    input_features_t: LogicalTensor
    feature_attention_mask_t: LogicalTensor
    position_ids_t: LogicalTensor
    audio_encoder_t: AudioEncoderTensors
    embed_tokens_t: EmbedTokensTensors
    audio_inject_t: AudioInjectTensors
    key_caches: tuple[LogicalTensor, ...]
    value_caches: tuple[LogicalTensor, ...]
    prefill_rope_t: RopeTableTensors
    decode_rope_t: RopeTableTensors
    text_layer_ts: tuple[TextLayerTensors, ...]
    text_norm_t: TextNormTensors
    lm_head_t: LmHeadTensors
    decode_embed_t: DecodeEmbedTensors
    decode_layer_ts: tuple[DecodeLayerTensors, ...]
    decode_norm_t: DecodeNormTensors
    decode_lm_head_t: DecodeLmHeadTensors
    eos_token_ids_t: LogicalTensor
    next_token_t: LogicalTensor
    done_t: LogicalTensor
    generated_tokens_t: LogicalTensor
    generated_length_t: LogicalTensor
    stopped_t: LogicalTensor
    token_index_t: LogicalTensor


_MODEL_TENSORS: ExportedQwen3AsrTensors | None = None


def create_model_tensors(
    *,
    input_ids_shape: tuple[int, ...],
    attention_mask_shape: tuple[int, ...],
    input_features_shape: tuple[int, ...],
    feature_attention_mask_shape: tuple[int, ...],
    prompt_length: int,
    max_sequence_length: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    max_new_tokens: int,
    eos_token_count: int,
) -> ExportedQwen3AsrTensors:
    input_ids_t = _host_input_tensor(
        "spike.text.prefill.input_ids",
        "int64",
        input_ids_shape,
    )
    attention_mask_t = _host_input_tensor(
        "spike.text.prefill.attention_mask",
        "int64",
        attention_mask_shape,
    )
    input_features_t = _host_input_tensor(
        "spike.text.prefill.input_features",
        "float32",
        input_features_shape,
    )
    feature_attention_mask_t = _host_input_tensor(
        "spike.text.prefill.feature_attention_mask",
        "int64",
        feature_attention_mask_shape,
    )
    position_ids_t = _host_input_tensor(
        "spike.text.prefill.position_ids",
        "int64",
        (3, 1, prompt_length),
    )

    audio_encoder_t = create_audio_encoder(
        "spike.audio",
        request_state_outputs={AUDIO_ENCODER_OUTPUT},
    )
    embed_tokens_t = create_embed_tokens(
        "spike.text.embed",
        input=input_ids_t,
    )
    audio_inject_t = create_audio_inject(
        "spike.text.audio_inject",
        audio_features=audio_encoder_t.linear_110,
        index_copy=embed_tokens_t.embedding,
    )
    key_caches = tuple(
        _request_state_tensor(
            f"spike.text.layers.{layer_idx}.key_cache",
            "float32",
            (1, num_key_value_heads, max_sequence_length, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for layer_idx in range(num_hidden_layers)
    )
    value_caches = tuple(
        _request_state_tensor(
            f"spike.text.layers.{layer_idx}.value_cache",
            "float32",
            (1, num_key_value_heads, max_sequence_length, head_dim),
            semantic=TensorSemantic.KV_CACHE,
        )
        for layer_idx in range(num_hidden_layers)
    )
    prefill_rope_t = create_rope_table(
        "spike.text.prefill.rope",
        batch=1,
        sequence_length=prompt_length,
        head_dim=head_dim,
    )
    decode_rope_t = create_rope_table(
        "spike.decode.rope",
        batch=1,
        sequence_length=1,
        head_dim=head_dim,
    )

    text_layer_ts_list: list[TextLayerTensors] = []
    text_hidden = audio_inject_t.index_copy
    for layer_idx in range(num_hidden_layers):
        layer_tensors = create_text_layer(
            f"spike.text.layer.{layer_idx}",
            layer_idx=layer_idx,
            hidden_states=text_hidden,
            index_copy=key_caches[layer_idx],
            index_copy_1=value_caches[layer_idx],
            position_embeddings_0=prefill_rope_t.cos,
            position_embeddings_1=prefill_rope_t.sin,
            cache_position=text_layer_ts_list[0].cache_position if layer_idx > 0 else None,
        )
        text_layer_ts_list.append(layer_tensors)
        text_hidden = layer_tensors.add_7
    text_layer_ts = tuple(text_layer_ts_list)

    text_norm_t = create_text_norm(
        "spike.text.norm",
        hidden_states=text_layer_ts[-1].add_7,
    )
    lm_head_t = create_lm_head(
        "spike.text.lm_head",
        input=text_norm_t.mul_1,
        request_state_outputs={LM_HEAD_OUTPUT},
    )
    decode_embed_t = create_decode_embed(
        "spike.decode.embed",
        p_weight=embed_tokens_t.p_weight,
    )

    decode_layer_ts_list: list[DecodeLayerTensors] = []
    decode_hidden = decode_embed_t.embedding
    for layer_idx, prefill_layer_tensors in enumerate(text_layer_ts):
        layer_tensors = create_decode_layer(
            f"spike.decode.layer.{layer_idx}",
            layer_idx=layer_idx,
            p_input_layernorm_weight=prefill_layer_tensors.p_input_layernorm_weight,
            p_post_attention_layernorm_weight=(
                prefill_layer_tensors.p_post_attention_layernorm_weight
            ),
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
            position_embeddings_0=decode_rope_t.cos,
            position_embeddings_1=decode_rope_t.sin,
            cache_position=decode_layer_ts_list[0].cache_position if layer_idx > 0 else None,
        )
        decode_layer_ts_list.append(layer_tensors)
        decode_hidden = layer_tensors.add_7
    decode_layer_ts = tuple(decode_layer_ts_list)

    decode_norm_t = create_decode_norm(
        "spike.decode.norm",
        p_weight=text_norm_t.p_weight,
        hidden_states=decode_layer_ts[-1].add_7,
    )
    decode_lm_head_t = create_decode_lm_head(
        "spike.decode.lm_head",
        p_weight=lm_head_t.p_weight,
        input=decode_norm_t.mul_1,
        request_state_outputs={DECODE_LM_HEAD_OUTPUT},
    )

    eos_token_ids_t = _host_input_tensor(
        "spike.token_select.eos_token_ids",
        "int64",
        (eos_token_count,),
    )
    next_token_t = _request_output_tensor("spike.token_select.next_token", "int64", (1,))
    done_t = _request_output_tensor("spike.token_select.done", "uint32", (1,))
    generated_tokens_t = _request_state_tensor(
        "spike.token_select.generated_tokens",
        "int64",
        (1, max_new_tokens),
        semantic=TensorSemantic.TOKEN,
    )
    generated_length_t = _request_state_tensor(
        "spike.token_select.generated_length",
        "uint32",
        (1,),
        semantic=TensorSemantic.TOKEN,
    )
    stopped_t = _request_state_tensor(
        "spike.token_select.stopped",
        "uint32",
        (1,),
        semantic=TensorSemantic.TOKEN,
    )
    token_index_t = _host_input_tensor("spike.token_select.token_index", "int64", (1,))

    global _MODEL_TENSORS
    _MODEL_TENSORS = ExportedQwen3AsrTensors(
        input_ids_t=input_ids_t,
        attention_mask_t=attention_mask_t,
        input_features_t=input_features_t,
        feature_attention_mask_t=feature_attention_mask_t,
        position_ids_t=position_ids_t,
        audio_encoder_t=audio_encoder_t,
        embed_tokens_t=embed_tokens_t,
        audio_inject_t=audio_inject_t,
        key_caches=key_caches,
        value_caches=value_caches,
        prefill_rope_t=prefill_rope_t,
        decode_rope_t=decode_rope_t,
        text_layer_ts=text_layer_ts,
        text_norm_t=text_norm_t,
        lm_head_t=lm_head_t,
        decode_embed_t=decode_embed_t,
        decode_layer_ts=decode_layer_ts,
        decode_norm_t=decode_norm_t,
        decode_lm_head_t=decode_lm_head_t,
        eos_token_ids_t=eos_token_ids_t,
        next_token_t=next_token_t,
        done_t=done_t,
        generated_tokens_t=generated_tokens_t,
        generated_length_t=generated_length_t,
        stopped_t=stopped_t,
        token_index_t=token_index_t,
    )
    return _MODEL_TENSORS


def model_tensors() -> ExportedQwen3AsrTensors:
    if _MODEL_TENSORS is None:
        raise RuntimeError("create_model_tensors must be called before generated dispatch")
    return _MODEL_TENSORS


def _host_input_tensor(name: str, dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.INPUT,
        memory=MemoryClass.HOST_INPUT,
        lifetime=TensorLifetime.FRAME,
    )


def _request_output_tensor(name: str, dtype: str, shape: tuple[int, ...]) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.OUTPUT,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
    )


def _request_state_tensor(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    *,
    semantic: TensorSemantic | None = None,
) -> LogicalTensor:
    return LogicalTensor(
        name=name,
        spec=TensorSpec(dtype=dtype, shape=shape),
        role=TensorRole.STATE,
        memory=MemoryClass.REQUEST_STATE,
        lifetime=TensorLifetime.REQUEST,
        semantic=semantic,
    )
