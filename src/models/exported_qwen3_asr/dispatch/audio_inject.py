"""Generated dispatch function for run_audio_inject."""

from __future__ import annotations

from models.exported_qwen3_asr.tensors.model import model_tensors
from models.exported_qwen3_asr.shaders.index_copy_f32_7ba4f1ff13 import INDEX_COPY_F32_7BA4F1FF13
from models.exported_qwen3_asr.tensors.audio_inject import AudioInjectTensors
from torch2vk.runtime.session import RuntimeSession


def _run_audio_inject_with_tensors(rt: RuntimeSession, tensors: AudioInjectTensors) -> None:
    INDEX_COPY_F32_7BA4F1FF13(rt, cache=tensors.index_copy, index=tensors.audio_positions, src=tensors.unsqueeze)


def run_audio_inject(rt: RuntimeSession) -> None:
    _run_audio_inject_with_tensors(rt, model_tensors().audio_inject)
