"""Generated dispatch function for run_audio_head."""

from __future__ import annotations

from models.quantized_omnivoice.tensors.model import model_tensors
from models.quantized_omnivoice.shaders.linear_nobias_q6_k_f32 import LINEAR_NOBIAS_Q6_K_F32
from models.quantized_omnivoice.tensors.audio_head import AudioHeadTensors
from torch2vk.runtime.session import RuntimeSession


def _run_audio_head_with_tensors(rt: RuntimeSession, tensors: AudioHeadTensors) -> None:
    LINEAR_NOBIAS_Q6_K_F32(rt, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def run_audio_head(rt: RuntimeSession) -> None:
    _run_audio_head_with_tensors(rt, model_tensors().audio_head)
