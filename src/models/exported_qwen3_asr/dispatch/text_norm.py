"""Generated dispatch function for run_text_norm."""

from __future__ import annotations

from models.exported_qwen3_asr.tensors.model import model_tensors
from models.exported_qwen3_asr.shaders.rms_norm_bf16w_f32 import RMS_NORM_BF16W_F32
from models.exported_qwen3_asr.tensors.text_norm import TextNormTensors
from torch2vk.runtime.session import RuntimeSession


def _run_text_norm_with_tensors(rt: RuntimeSession, tensors: TextNormTensors) -> None:
    RMS_NORM_BF16W_F32(rt, x=tensors.hidden_states, weight=tensors.p_weight, output=tensors.rms_norm)


def run_text_norm(rt: RuntimeSession) -> None:
    tensors = model_tensors().text_norm
    _run_text_norm_with_tensors(rt, tensors)
