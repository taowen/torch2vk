"""Generated dispatch function for run_flux_join."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.flux_join_cat_2_f32 import FLUX_JOIN_CAT_2_F32
from models.quantized_klein9b.tensors.flux_join import FluxJoinTensors
from torch2vk.runtime.session import RuntimeSession


def _run_flux_join_with_tensors(rt: RuntimeSession, tensors: FluxJoinTensors) -> None:
    FLUX_JOIN_CAT_2_F32(rt, x0=tensors.txt, x1=tensors.img, output=tensors.cat)


def run_flux_join(rt: RuntimeSession) -> None:
    tensors = model_tensors().flux_join
    _run_flux_join_with_tensors(rt, tensors)
