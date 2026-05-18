from __future__ import annotations

import types
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_RMS_NORM_CLASS_NAMES = frozenset(
    (
        "Qwen3RMSNorm",
        "Qwen3ASRTextRMSNorm",
        "Qwen3ASRThinkerTextRMSNorm",
    )
)


class _RMSNormModule(Protocol):
    weight: Tensor
    variance_epsilon: float


def patch_rms_norm_modules(module: nn.Module) -> None:
    for submodule in module.modules():
        if submodule.__class__.__name__ not in _RMS_NORM_CLASS_NAMES:
            continue
        rms_norm = cast(_RMSNormModule, submodule)
        setattr(submodule, "forward", types.MethodType(_rms_norm_forward, rms_norm))


def _rms_norm_forward(self: _RMSNormModule, hidden_states: Tensor) -> Tensor:
    return F.rms_norm(
        hidden_states,
        (self.weight.shape[0],),
        self.weight,
        eps=self.variance_epsilon,
    )
