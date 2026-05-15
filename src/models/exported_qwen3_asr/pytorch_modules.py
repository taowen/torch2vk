"""PyTorch modules shared by exported_qwen3_asr export and reference compare."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.modeling_outputs import BaseModelOutput


class AudioInjectModule(torch.nn.Module):
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        audio_positions: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.index_copy(
            inputs_embeds,
            1,
            audio_positions,
            audio_features.unsqueeze(0),
        )


def get_feat_extract_output_lengths(input_lengths: np.ndarray) -> np.ndarray:
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def compute_positional_embedding(length: int, channels: int) -> np.ndarray:
    max_timescale = 10000.0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = np.exp(-log_timescale_increment * np.arange(channels // 2, dtype=np.float32))
    scaled_time = np.arange(length, dtype=np.float32)[:, None] * inv_timescales[None, :]
    return np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1).astype(np.float32)


def preprocess_audio_inputs(
    input_ids: np.ndarray,
    input_features: np.ndarray,
    feature_attention_mask: np.ndarray,
    *,
    position_embedding_shape: tuple[int, ...],
    d_model: int,
) -> dict[str, np.ndarray]:
    feat_len = int(np.asarray(feature_attention_mask).sum(axis=-1).reshape(-1)[0])
    mel = np.ascontiguousarray(np.asarray(input_features)[0, :, :feat_len], dtype=np.float32)

    n_window = 50
    chunk_num = int(np.ceil(feat_len / (n_window * 2)))
    chunk_lengths = np.full(chunk_num, n_window * 2, dtype=np.int64)
    remainder = feat_len % (n_window * 2)
    if remainder != 0:
        chunk_lengths[-1] = remainder

    features_t = mel.T
    chunks = []
    offset = 0
    for chunk_length in chunk_lengths:
        chunk = features_t[offset : offset + chunk_length]
        chunks.append(chunk)
        offset += int(chunk_length)

    max_chunk_len = int(chunk_lengths.max())
    num_mel = mel.shape[0]
    padded_feature = np.zeros((chunk_num, 1, num_mel, max_chunk_len), dtype=np.float32)
    for index, chunk in enumerate(chunks):
        padded_feature[index, 0, :, : chunk.shape[0]] = chunk.T

    _, t_dim, _ = position_embedding_shape
    pos_emb = compute_positional_embedding(t_dim, d_model)
    position_embedding = np.ascontiguousarray(
        np.broadcast_to(pos_emb[None, :t_dim, :], position_embedding_shape),
        dtype=np.float32,
    )

    feature_lens_after_cnn = get_feat_extract_output_lengths(chunk_lengths)
    valid_positions = [
        chunk_index * t_dim + offset
        for chunk_index, feature_len in enumerate(feature_lens_after_cnn)
        for offset in range(int(feature_len))
    ]
    compact_index = np.array(valid_positions, dtype=np.int64)

    n_window_infer = 800
    aftercnn_lens = get_feat_extract_output_lengths(np.array([feat_len], dtype=np.int64))
    window_aftercnn = int(feature_lens_after_cnn.max()) * (n_window_infer // (n_window * 2))
    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens:
        cnn_len = int(cnn_len)
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        rem = cnn_len % window_aftercnn
        if rem != 0:
            cu_chunk_lens += [rem]
    cu_seqlens = np.cumsum(cu_chunk_lens, dtype=np.int32)

    seq_len = compact_index.shape[0]
    attention_mask = np.full(
        (1, 1, seq_len, seq_len),
        -np.finfo(np.float32).max,
        dtype=np.float32,
    )
    for index in range(1, len(cu_seqlens)):
        start, end = int(cu_seqlens[index - 1]), int(cu_seqlens[index])
        attention_mask[0, 0, start:end, start:end] = 0.0

    audio_positions = np.where(np.asarray(input_ids).reshape(-1) == 151676)[0].astype(np.int64)
    return {
        "padded_feature": padded_feature,
        "position_embedding": position_embedding,
        "compact_index": compact_index,
        "audio_attention_mask": attention_mask,
        "cu_seqlens": cu_seqlens,
        "audio_positions": audio_positions,
    }


def audio_position_embedding_shape(
    *,
    feature_length: int,
    d_model: int,
) -> tuple[int, int, int]:
    n_window = 50
    chunk_num = int(np.ceil(feature_length / (n_window * 2)))
    chunk_lengths = np.full(chunk_num, n_window * 2, dtype=np.int64)
    remainder = feature_length % (n_window * 2)
    if remainder != 0:
        chunk_lengths[-1] = remainder
    time = int(get_feat_extract_output_lengths(chunk_lengths).max())
    return (chunk_num, time, d_model)


def torch_tensor(inputs: dict[str, np.ndarray], name: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(inputs[name])).cuda()


class AudioEncoderModule(torch.nn.Module):
    def __init__(self, audio_tower: torch.nn.Module) -> None:
        super().__init__()
        self.audio_tower = audio_tower

    def forward(
        self,
        x: torch.Tensor | None = None,
        position_embedding: torch.Tensor | None = None,
        compact_index: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_lens: torch.Tensor | None = None,
        aftercnn_lens: torch.Tensor | None = None,
    ) -> BaseModelOutput:
        del aftercnn_lens
        if position_embedding is not None and compact_index is not None and attention_mask is not None:
            return self._forward_preprocessed(
                x=_require_tensor(x, "x"),
                position_embedding=position_embedding,
                compact_index=compact_index,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            )
        raw_input_features = input_features if input_features is not None else x
        if raw_input_features is None or feature_lens is None:
            raise ValueError(
                "AudioEncoderModule requires either preprocessed audio tensors or "
                "raw input_features with feature_lens"
            )
        return self._forward_raw(raw_input_features, feature_lens)

    def _forward_preprocessed(
        self,
        *,
        x: torch.Tensor,
        position_embedding: torch.Tensor,
        compact_index: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> BaseModelOutput:
        last_hidden_state = _run_audio_tower(
            self.audio_tower,
            x,
            position_embedding,
            compact_index,
            cu_seqlens,
            attention_mask,
        )
        return BaseModelOutput(
            last_hidden_state=cast(torch.FloatTensor, last_hidden_state)
        )

    def _forward_raw(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> BaseModelOutput:
        config = getattr(self.audio_tower, "config")
        d_model = int(getattr(config, "d_model"))
        feat_len = int(feature_lens.detach().cpu().reshape(-1)[0].item())
        features = input_features.detach().cpu().float().numpy()
        if features.ndim == 2:
            batched_features = features[None, :, :feat_len]
        elif features.ndim == 3:
            batched_features = features[:, :, :feat_len]
        else:
            raise ValueError(f"input_features must be 2D or 3D, got shape {features.shape}")
        arrays = preprocess_audio_inputs(
            np.zeros((1, 0), dtype=np.int64),
            batched_features,
            np.ones((1, feat_len), dtype=np.int64),
            position_embedding_shape=audio_position_embedding_shape(
                feature_length=feat_len,
                d_model=d_model,
            ),
            d_model=d_model,
        )
        device = input_features.device
        dtype = input_features.dtype
        return self._forward_preprocessed(
            x=torch.from_numpy(arrays["padded_feature"]).to(device=device, dtype=dtype),
            position_embedding=torch.from_numpy(arrays["position_embedding"]).to(
                device=device,
                dtype=dtype,
            ),
            compact_index=torch.from_numpy(arrays["compact_index"]).to(device=device),
            cu_seqlens=torch.from_numpy(arrays["cu_seqlens"]).to(device=device),
            attention_mask=torch.from_numpy(arrays["audio_attention_mask"]).to(
                device=device,
                dtype=dtype,
            ),
        )


class AudioEncoderReference:
    def __init__(self, audio_tower: torch.nn.Module) -> None:
        self.audio_encoder = AudioEncoderModule(audio_tower).float().cuda().eval()

    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        with torch.no_grad():
            output = self.audio_encoder(
                x=torch_tensor(inputs, "x").float(),
                position_embedding=torch_tensor(inputs, "position_embedding").float(),
                compact_index=torch_tensor(inputs, "compact_index").long(),
                attention_mask=torch_tensor(inputs, "attention_mask").float(),
            )
        return {"linear_110": output.last_hidden_state}


class AudioInjectReference:
    def __init__(self) -> None:
        self.audio_inject = AudioInjectModule().cuda().eval()

    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        with torch.no_grad():
            output = self.audio_inject(
                torch_tensor(inputs, "inputs_embeds").float(),
                torch_tensor(inputs, "audio_positions").long(),
                torch_tensor(inputs, "audio_features").float(),
            )
        return {"embedding": output}


class TextReferenceState:
    def __init__(self, thinker: torch.nn.Module) -> None:
        self.thinker = thinker
        self.text_model = cast(torch.nn.Module, thinker.get_submodule("model"))
        layers = cast(torch.nn.ModuleList, self.text_model.get_submodule("layers"))
        self.layers = tuple(cast(torch.nn.Module, layer) for layer in layers)


class TextLayerReference:
    def __init__(self, state: TextReferenceState, layer_idx: int, *, prefill: bool) -> None:
        self.state = state
        self.layer = state.layers[layer_idx]
        self.prefill = prefill

    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        hidden = torch_tensor(inputs, "hidden_states").float()
        cos = torch_tensor(inputs, "position_embeddings_0").float()
        sin = torch_tensor(inputs, "position_embeddings_1").float()
        cache_position = torch_tensor(inputs, "cache_position").long()
        key_cache = torch_tensor(inputs, "key_cache")
        value_cache = torch_tensor(inputs, "value_cache")
        with torch.no_grad():
            output, updated_key_cache, updated_value_cache = _run_text_layer_with_explicit_cache(
                self.layer,
                hidden_states=hidden,
                position_embeddings=(cos, sin),
                key_cache=key_cache,
                value_cache=value_cache,
                cache_position=cache_position,
                prefill=self.prefill,
            )
        return {
            "add_7": output,
            "index_copy": updated_key_cache,
            "index_copy_1": updated_value_cache,
        }


class TokenSelectReference:
    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        logits = np.asarray(inputs["logits"])
        eos_token_ids = set(int(token) for token in np.asarray(inputs["eos_token_ids"]).reshape(-1))
        token = int(np.argmax(logits[0, -1, :]))
        return {
            "next_token": np.array([[token]], dtype=np.int64),
            "done": np.array([1 if token in eos_token_ids else 0], dtype=np.uint32),
        }


class TokenStoreReference:
    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        generated_tokens = np.asarray(inputs["generated_tokens"]).copy()
        generated_length = np.asarray(inputs["generated_length"]).copy()
        stopped = np.asarray(inputs["stopped"]).copy()
        index = int(np.asarray(inputs["token_index"]).reshape(-1)[0])
        if index < generated_tokens.shape[1] and int(stopped[0]) == 0:
            generated_tokens[0, index] = int(np.asarray(inputs["next_token"]).reshape(-1)[0])
            generated_length[0] = index + 1
            if int(np.asarray(inputs["done"]).reshape(-1)[0]) != 0:
                stopped[0] = 1
        return {
            "generated_tokens": generated_tokens,
            "generated_length": generated_length,
            "stopped": stopped,
        }


def _causal_attention_mask(hidden: torch.Tensor) -> torch.Tensor:
    sequence_length = int(hidden.shape[1])
    min_value = torch.finfo(hidden.dtype).min
    return torch.triu(
        torch.full(
            (1, 1, sequence_length, sequence_length),
            min_value,
            dtype=hidden.dtype,
            device=hidden.device,
        ),
        diagonal=1,
    )


def _run_text_layer_with_explicit_cache(
    layer: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    prefill: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = hidden_states
    normed = layer.get_submodule("input_layernorm")(hidden_states)
    attention_output, updated_key_cache, updated_value_cache = _run_text_attention_with_explicit_cache(
        layer.get_submodule("self_attn"),
        hidden_states=normed,
        position_embeddings=position_embeddings,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        prefill=prefill,
    )
    hidden_states = residual + attention_output

    residual = hidden_states
    hidden_states = layer.get_submodule("post_attention_layernorm")(hidden_states)
    hidden_states = layer.get_submodule("mlp")(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, updated_key_cache, updated_value_cache


def _run_text_attention_with_explicit_cache(
    attention: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    prefill: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    head_dim = int(getattr(attention, "head_dim"))
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = attention.get_submodule("q_proj")(hidden_states).view(hidden_shape)
    query_states = attention.get_submodule("q_norm")(query_states).transpose(1, 2)
    key_states = attention.get_submodule("k_proj")(hidden_states).view(hidden_shape)
    key_states = attention.get_submodule("k_norm")(key_states).transpose(1, 2)
    value_states = attention.get_submodule("v_proj")(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    updated_key_cache = key_cache.clone()
    updated_value_cache = value_cache.clone()
    positions = cache_position.to(device=updated_key_cache.device)
    updated_key_cache.index_copy_(2, positions, key_states.to(dtype=updated_key_cache.dtype))
    updated_value_cache.index_copy_(2, positions, value_states.to(dtype=updated_value_cache.dtype))

    if prefill:
        attention_key = key_states
        attention_value = value_states
        attention_mask = _causal_attention_mask(hidden_states)
    else:
        cache_length = int(positions.max().item()) + 1
        attention_key = updated_key_cache[:, :, :cache_length, :].to(dtype=query_states.dtype)
        attention_value = updated_value_cache[:, :, :cache_length, :].to(dtype=query_states.dtype)
        attention_mask = None

    attention_output, _ = eager_attention_forward(
        attention,
        query_states,
        attention_key,
        attention_value,
        attention_mask,
        dropout=0.0,
        scaling=float(getattr(attention, "scaling")),
    )
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    attention_output = attention.get_submodule("o_proj")(attention_output)
    return attention_output, updated_key_cache, updated_value_cache


def _require_tensor(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"AudioEncoderModule requires {name}")
    return value


def _run_audio_tower(
    audio_tower: torch.nn.Module,
    x: torch.Tensor,
    position_embedding: torch.Tensor,
    compact_index: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    x = torch.nn.functional.gelu(audio_tower.get_submodule("conv2d1")(x))
    x = torch.nn.functional.gelu(audio_tower.get_submodule("conv2d2")(x))
    x = torch.nn.functional.gelu(audio_tower.get_submodule("conv2d3")(x))
    batch, channels, freq, time = x.shape
    hidden_states = audio_tower.get_submodule("conv_out")(
        x.reshape(batch, channels * freq, time).transpose(1, 2)
    )
    hidden_states = hidden_states + position_embedding
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    hidden_states = torch.index_select(hidden_states, 0, compact_index)

    for layer in cast(torch.nn.ModuleList, audio_tower.get_submodule("layers")):
        if cu_seqlens is None:
            hidden_states = _run_audio_layer(layer, hidden_states, attention_mask)
        else:
            hidden_states = layer(
                hidden_states,
                cu_seqlens,
                attention_mask=attention_mask,
            )[0]

    hidden_states = audio_tower.get_submodule("ln_post")(hidden_states)
    hidden_states = audio_tower.get_submodule("proj1")(hidden_states)
    hidden_states = cast(torch.nn.Module, getattr(audio_tower, "act"))(hidden_states)
    return audio_tower.get_submodule("proj2")(hidden_states)


def _run_audio_layer(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = layer.get_submodule("self_attn_layer_norm")(hidden_states)
    hidden_states = _run_audio_attention(
        layer.get_submodule("self_attn"),
        hidden_states,
        attention_mask,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = layer.get_submodule("final_layer_norm")(hidden_states)
    hidden_states = layer.get_submodule("fc1")(hidden_states)
    hidden_states = cast(torch.nn.Module, getattr(layer, "activation_fn"))(hidden_states)
    hidden_states = layer.get_submodule("fc2")(hidden_states)
    return residual + hidden_states


def _run_audio_attention(
    attention: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    seq_length, _ = hidden_states.size()
    num_heads = int(getattr(attention, "num_heads"))
    query_states = attention.get_submodule("q_proj")(hidden_states).reshape(
        seq_length, num_heads, -1
    )
    key_states = attention.get_submodule("k_proj")(hidden_states).reshape(
        seq_length, num_heads, -1
    )
    value_states = attention.get_submodule("v_proj")(hidden_states).reshape(
        seq_length, num_heads, -1
    )

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    attention_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(getattr(attention, "scaling")),
    )
    attention_output = attention_output.transpose(1, 2).reshape(seq_length, -1)
    return attention.get_submodule("out_proj")(attention_output)
