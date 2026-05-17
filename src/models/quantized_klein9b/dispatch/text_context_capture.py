"""Generated dispatch function for run_text_context_capture."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.cast_f16_to_f32_a517f4d3a3 import CAST_F16_TO_F32_A517F4D3A3
from models.quantized_klein9b.shaders.cat_3_f32 import CAT_3_F32
from models.quantized_klein9b.tensors.text_context_capture import TextContextCaptureTensors
from torch2vk.runtime.session import RuntimeSession


def _run_text_context_capture_with_tensors(rt: RuntimeSession, tensors: TextContextCaptureTensors) -> None:
    CAST_F16_TO_F32_A517F4D3A3(rt, x=tensors.layer_9, output=tensors.to)
    CAST_F16_TO_F32_A517F4D3A3(rt, x=tensors.layer_18, output=tensors.to_1)
    CAST_F16_TO_F32_A517F4D3A3(rt, x=tensors.layer_27, output=tensors.to_2)
    CAT_3_F32(rt, x0=tensors.to, x1=tensors.to_1, x2=tensors.to_2, output=tensors.cat)
    rt.release_frame_workspace(tensors.to)
    rt.release_frame_workspace(tensors.to_1)
    rt.release_frame_workspace(tensors.to_2)


def run_text_context_capture(rt: RuntimeSession) -> None:
    tensors = model_tensors().text_context_capture
    _run_text_context_capture_with_tensors(rt, tensors)
