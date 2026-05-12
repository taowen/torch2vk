"""Generated dispatch function for run_embed_tokens."""

from __future__ import annotations

from models.optimized_qwen3.tensors.model import model_tensors
from models.optimized_qwen3.shaders.embedding_q4_k_f32 import EMBEDDING_Q4_K_F32
from models.optimized_qwen3.tensors.embed_tokens import EmbedTokensTensors
from torch2vk.runtime.session import RuntimeSession


def _run_embed_tokens_with_tensors(rt: RuntimeSession, tensors: EmbedTokensTensors) -> None:
    EMBEDDING_Q4_K_F32(rt, weight=tensors.p_weight, indices=tensors.input, output=tensors.embedding)


def run_embed_tokens(rt: RuntimeSession) -> None:
    _run_embed_tokens_with_tensors(rt, model_tensors().embed_tokens)
