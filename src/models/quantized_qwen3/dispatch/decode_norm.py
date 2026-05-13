"""Generated dispatch function for run_decode_norm."""

from __future__ import annotations

from models.quantized_qwen3.tensors.model import model_tensors
from models.quantized_qwen3.shaders.rms_norm_mul_f16_f32 import RMS_NORM_MUL_F16_F32
from models.quantized_qwen3.tensors.decode_norm import DecodeNormTensors
from torch2vk.runtime.session import RuntimeSession


def _run_decode_norm_with_tensors(rt: RuntimeSession, tensors: DecodeNormTensors) -> None:
    RMS_NORM_MUL_F16_F32(rt, x=tensors.to, weight=tensors.p_weight, output=tensors.mul_1)


def run_decode_norm(rt: RuntimeSession) -> None:
    _run_decode_norm_with_tensors(rt, model_tensors().decode_norm)
