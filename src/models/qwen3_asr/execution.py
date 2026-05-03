"""Qwen3-ASR eager runtime entry points."""

from __future__ import annotations

from models.qwen3_asr.position_embedding import (
    Qwen3AsrPositionEmbeddingOutput,
    run_qwen3_asr_position_embedding_frame,
)
from models.qwen3_asr.tensors.position_embedding import Qwen3AsrPositionEmbeddingTensors
from torch2vk.runtime.session import RuntimeSession


def run_qwen3_asr_audio_tower_position_embedding(
    rt: RuntimeSession,
    tensors: Qwen3AsrPositionEmbeddingTensors,
    *,
    pytorch_model: object,
    seqlen: int,
) -> Qwen3AsrPositionEmbeddingOutput:
    return run_qwen3_asr_position_embedding_frame(
        rt,
        tensors,
        pytorch_model=pytorch_model,
        seqlen=seqlen,
    )
