"""Generated dispatch function for run_text_embed."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.embedding_q8_0_f32 import EMBEDDING_Q8_0_F32
from models.quantized_klein9b.tensors.embed_tokens import EmbedTokensTensors
from torch2vk.runtime.session import RuntimeSession


def _run_text_embed_with_tensors(rt: RuntimeSession, tensors: EmbedTokensTensors) -> None:
    EMBEDDING_Q8_0_F32(rt, weight=tensors.p_weight, indices=tensors.input, output=tensors.embedding)


def run_text_embed(rt: RuntimeSession) -> None:
    tensors = model_tensors().text_embed
    _run_text_embed_with_tensors(rt, tensors)
