"""Generated dispatch function for run_decode_norm."""

from __future__ import annotations

from models.optimized_qwen3_asr.tensors.model import model_tensors
from models.optimized_qwen3_asr.shaders.decode_rms_norm_f32 import DECODE_RMS_NORM_F32
from models.optimized_qwen3_asr.tensors.decode_norm import DecodeNormTensors
from torch2vk.runtime.session import RuntimeSession


def _run_decode_norm_with_tensors(rt: RuntimeSession, tensors: DecodeNormTensors) -> None:
    DECODE_RMS_NORM_F32(rt, x=tensors.to, weight=tensors.p_weight, output=tensors.mul_1)


def run_decode_norm(rt: RuntimeSession) -> None:
    _run_decode_norm_with_tensors(rt, model_tensors().decode_norm)
