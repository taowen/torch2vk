"""Generated dispatch function for run_decode_embed."""

from __future__ import annotations

from models.quantized_qwen3_asr.tensors.model import model_tensors
from models.quantized_qwen3_asr.shaders.embedding_q8_0_f32 import EMBEDDING_Q8_0_F32
from models.quantized_qwen3_asr.tensors.decode_embed import DecodeEmbedTensors
from torch2vk.runtime.session import RuntimeSession


def _run_decode_embed_with_tensors(rt: RuntimeSession, tensors: DecodeEmbedTensors) -> None:
    EMBEDDING_Q8_0_F32(rt, weight=tensors.p_weight, indices=tensors.input, output=tensors.embedding)


def run_decode_embed(rt: RuntimeSession) -> None:
    _run_decode_embed_with_tensors(rt, model_tensors().decode_embed)
