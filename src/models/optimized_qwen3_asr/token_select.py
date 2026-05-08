"""Qwen3-ASR token selection frame."""

from __future__ import annotations

from models.optimized_qwen3_asr.shaders.token_select_f32 import QWEN3_ASR_TOKEN_SELECT_GREEDY_F32
from models.optimized_qwen3_asr.tensors.text import Qwen3AsrTokenSelectTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.session import RuntimeSession


def run_qwen3_asr_token_select(
    rt: RuntimeSession,
    tensors: Qwen3AsrTokenSelectTensors,
    *,
    logits: LogicalTensor,
) -> LogicalTensor:
    """Select the greedy next token from a single logits row and set EOS done state."""
    with rt.frame("qwen3_asr.token_select"):
        QWEN3_ASR_TOKEN_SELECT_GREEDY_F32(
            rt,
            logits=logits,
            eos_token_ids=tensors.eos_token_ids,
            next_token=tensors.next_token,
            done=tensors.done,
        )
    return tensors.next_token
