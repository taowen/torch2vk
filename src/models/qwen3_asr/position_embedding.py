"""Qwen3-ASR audio position embedding frame."""

from __future__ import annotations

from dataclasses import dataclass

from models.qwen3_asr.shaders.position_embedding_f32 import QWEN3_ASR_POSITION_EMBEDDING_F32
from models.qwen3_asr.tensors.position_embedding import Qwen3AsrPositionEmbeddingTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession


@dataclass(frozen=True, slots=True)
class Qwen3AsrPositionEmbeddingOutput:
    output: LogicalTensor


def run_qwen3_asr_position_embedding_frame(
    rt: RuntimeSession,
    tensors: Qwen3AsrPositionEmbeddingTensors,
    *,
    pytorch_model: object,
    seqlen: int,
) -> Qwen3AsrPositionEmbeddingOutput:
    with rt.frame(
        "qwen3_asr.audio_tower.position_embedding",
        scope={"seqlen": seqlen, "channels": tensors.output.spec.shape[1]},
        pytorch_model=pytorch_model,
        pytorch_args=(seqlen,),
    ):
        QWEN3_ASR_POSITION_EMBEDDING_F32(rt, output=tensors.output)
    return Qwen3AsrPositionEmbeddingOutput(output=tensors.output)
