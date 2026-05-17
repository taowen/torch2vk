"""Generated dispatch function for run_ae_entry."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.cast_f32_to_f16_c3ea119b28 import CAST_F32_TO_F16_C3EA119B28
from models.quantized_klein9b.shaders.permute_f32_43fee4ac2b import PERMUTE_F32_43FEE4AC2B
from models.quantized_klein9b.tensors.ae_entry import AEEntryTensors
from torch2vk.runtime.session import RuntimeSession


def _run_ae_entry_with_tensors(rt: RuntimeSession, tensors: AEEntryTensors) -> None:
    CAST_F32_TO_F16_C3EA119B28(rt, x=tensors.tokens, output=tensors.to)
    PERMUTE_F32_43FEE4AC2B(rt, x=tensors.to, output=tensors.permute)


def _require_ae_entry_tensors() -> AEEntryTensors:
    tensors = model_tensors().ae_entry
    if tensors is None:
        raise RuntimeError("AE entry tensors were not created")
    return tensors


def run_ae_entry(rt: RuntimeSession) -> None:
    tensors = _require_ae_entry_tensors()
    _run_ae_entry_with_tensors(rt, tensors)
