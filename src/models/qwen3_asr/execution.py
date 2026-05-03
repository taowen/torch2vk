"""Qwen3-ASR eager runtime entry points."""

from __future__ import annotations

from models.qwen3_asr.audio_tower import (
    Qwen3AsrAudioTowerConv2d1Output,
    run_qwen3_asr_audio_tower_conv2d1_frame,
)
from models.qwen3_asr.position_embedding import (
    Qwen3AsrPositionEmbeddingOutput,
    run_qwen3_asr_position_embedding_frame,
)
from models.qwen3_asr.tensors.audio_tower import Qwen3AsrAudioTowerConv2d1Tensors
from models.qwen3_asr.tensors.position_embedding import Qwen3AsrPositionEmbeddingTensors
from torch2vk.runtime.session import RuntimeSession


def run_qwen3_asr_audio_tower_conv2d1(
    rt: RuntimeSession,
    tensors: Qwen3AsrAudioTowerConv2d1Tensors,
    *,
    pytorch_model: object,
    pytorch_input: object,
) -> Qwen3AsrAudioTowerConv2d1Output:
    return run_qwen3_asr_audio_tower_conv2d1_frame(
        rt,
        tensors,
        pytorch_model=pytorch_model,
        pytorch_input=pytorch_input,
    )


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
