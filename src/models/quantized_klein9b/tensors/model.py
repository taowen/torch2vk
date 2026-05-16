"""Generated model-level tensor wiring for FLUX.2 Klein 9B."""

from __future__ import annotations

from dataclasses import dataclass

from models.quantized_klein9b.tensors.flux import FLUX_OUTPUT, FluxTensors, create_flux
from torch2vk.runtime.logical import LogicalTensor


@dataclass(frozen=True, slots=True)
class QuantizedKlein9BTensors:
    flux: FluxTensors


_MODEL_TENSORS: QuantizedKlein9BTensors | None = None


def create_model_tensors(
    *,
    image_seq_len: int = 1024,
    text_seq_len: int = 512,
) -> QuantizedKlein9BTensors:
    global _MODEL_TENSORS
    _MODEL_TENSORS = QuantizedKlein9BTensors(
        flux=create_flux(
            "klein9b.flux",
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
            request_state_outputs=frozenset((FLUX_OUTPUT,)),
        )
    )
    return _MODEL_TENSORS


def model_tensors() -> QuantizedKlein9BTensors:
    if _MODEL_TENSORS is None:
        raise RuntimeError("create_model_tensors() must be called before model_tensors()")
    return _MODEL_TENSORS


def flux_output(tensors: QuantizedKlein9BTensors | None = None) -> LogicalTensor:
    resolved = model_tensors() if tensors is None else tensors
    return getattr(resolved.flux, FLUX_OUTPUT)
