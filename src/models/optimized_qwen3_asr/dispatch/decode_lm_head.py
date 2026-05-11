"""Generated dispatch function for run_decode_lm_head."""

from __future__ import annotations

from models.optimized_qwen3_asr.tensors.model import model_tensors
from models.optimized_qwen3_asr.shaders.linear_nobias_q4_k_matvec_f32 import LINEAR_NOBIAS_Q4_K_MATVEC_F32
from models.optimized_qwen3_asr.tensors.decode_lm_head import DecodeLmHeadTensors
from torch2vk.runtime.session import RuntimeSession


def _run_decode_lm_head_with_tensors(rt: RuntimeSession, tensors: DecodeLmHeadTensors) -> None:
    LINEAR_NOBIAS_Q4_K_MATVEC_F32(rt, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def run_decode_lm_head(rt: RuntimeSession) -> None:
    _run_decode_lm_head_with_tensors(rt, model_tensors().decode_lm_head)
