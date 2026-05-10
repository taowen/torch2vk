"""Generated dispatch function for run_lm_head."""

from __future__ import annotations

from models.exported_qwen3_asr.tensors.model import model_tensors
from models.exported_qwen3_asr.shaders.lm_head_linear_nobias_f32 import LM_HEAD_LINEAR_NOBIAS_F32
from models.exported_qwen3_asr.tensors.lm_head import LmHeadTensors
from torch2vk.runtime.session import RuntimeSession


def _run_lm_head_with_tensors(rt: RuntimeSession, tensors: LmHeadTensors) -> None:
    LM_HEAD_LINEAR_NOBIAS_F32(rt, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def run_lm_head(rt: RuntimeSession) -> None:
    _run_lm_head_with_tensors(rt, model_tensors().lm_head)
