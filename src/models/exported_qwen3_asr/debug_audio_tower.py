"""Audio tower wrapper shared by exported_qwen3_asr export and PyTorch debug."""

from __future__ import annotations

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput
from typing import cast

from models.exported_qwen3_asr.export_forwards import export_audio_tower_forward


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


class DebugAudioTower(torch.nn.Module):
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
                "DebugAudioTower requires either preprocessed audio tensors or "
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
        output = export_audio_tower_forward(
            self.audio_tower,
            x,
            position_embedding,
            compact_index,
            cu_seqlens,
            attention_mask,
        )
        return BaseModelOutput(
            last_hidden_state=cast(torch.FloatTensor, output["last_hidden_state"])
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


def _require_tensor(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"DebugAudioTower requires {name}")
    return value
