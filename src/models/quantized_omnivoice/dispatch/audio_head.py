"""Generated dispatch function for run_audio_head."""

from __future__ import annotations

from models.quantized_omnivoice.tensors.model import model_tensors
from models.quantized_omnivoice.shaders.linear_nobias_q4_k_f32 import LINEAR_NOBIAS_Q4_K_F32
from models.quantized_omnivoice.shaders.linear_nobias_q6_k_f32 import LINEAR_NOBIAS_Q6_K_F32
from models.quantized_omnivoice.shaders.linear_nobias_q8_0_f32 import LINEAR_NOBIAS_Q8_0_F32
from models.quantized_omnivoice.tensors.audio_head import AudioHeadTensors
from torch2vk.runtime.quantized_dispatch import run_quantized_linear
from torch2vk.runtime.session import RuntimeSession


def _run_audio_head_with_tensors(rt: RuntimeSession, tensors: AudioHeadTensors) -> None:
    run_quantized_linear(rt, q4=LINEAR_NOBIAS_Q4_K_F32, q6=LINEAR_NOBIAS_Q6_K_F32, q8=LINEAR_NOBIAS_Q8_0_F32, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def run_audio_head(rt: RuntimeSession) -> None:
    _run_audio_head_with_tensors(rt, model_tensors().audio_head)
