"""Generated dispatch function for run_flux_pe_join."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.flux_pe_join_cat_2_f32 import FLUX_PE_JOIN_CAT_2_F32
from models.quantized_klein9b.tensors.flux_pe_join import FluxPeJoinTensors
from torch2vk.runtime.session import RuntimeSession


def _run_flux_pe_join_with_tensors(rt: RuntimeSession, tensors: FluxPeJoinTensors) -> None:
    FLUX_PE_JOIN_CAT_2_F32(rt, x0=tensors.pe_ctx, x1=tensors.pe_x, output=tensors.cat)


def run_flux_pe_join(rt: RuntimeSession) -> None:
    tensors = model_tensors().flux_pe_join
    _run_flux_pe_join_with_tensors(rt, tensors)
