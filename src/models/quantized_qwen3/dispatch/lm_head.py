"""Generated dispatch function for run_lm_head."""

from __future__ import annotations

from models.quantized_qwen3.tensors.model import model_tensors
from models.quantized_qwen3.shaders.linear_nobias_q4_k_f32 import LINEAR_NOBIAS_Q4_K_F32
from models.quantized_qwen3.tensors.lm_head import LmHeadTensors
from torch2vk.runtime.session import RuntimeSession


def _run_lm_head_with_tensors(rt: RuntimeSession, tensors: LmHeadTensors) -> None:
    LINEAR_NOBIAS_Q4_K_F32(rt, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def run_lm_head(rt: RuntimeSession) -> None:
    _run_lm_head_with_tensors(rt, model_tensors().lm_head)
