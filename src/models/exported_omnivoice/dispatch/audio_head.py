"""Generated dispatch function for run_audio_head."""

from __future__ import annotations

from models.exported_omnivoice.tensors.model import model_tensors
from models.exported_omnivoice.shaders.linear_nobias_f32w_f32 import LINEAR_NOBIAS_F32W_F32
from models.exported_omnivoice.tensors.audio_head import AudioHeadTensors
from torch2vk.runtime.session import RuntimeSession


def _run_audio_head_with_tensors(rt: RuntimeSession, tensors: AudioHeadTensors) -> None:
    LINEAR_NOBIAS_F32W_F32(rt, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def run_audio_head(rt: RuntimeSession) -> None:
    _run_audio_head_with_tensors(rt, model_tensors().audio_head)
