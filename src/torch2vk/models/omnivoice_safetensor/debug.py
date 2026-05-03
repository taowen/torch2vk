"""OmniVoice debug compatibility helpers."""

from __future__ import annotations

from torch2vk.logical import LogicalTensor

from .tensors.stage0 import OmniVoiceStage0Tensors
from .tensors.weights import OmniVoiceWeights, omnivoice_weight_tensors


def omnivoice_debug_initial_tensors(
    *,
    tensors: OmniVoiceStage0Tensors,
    weights: OmniVoiceWeights,
) -> tuple[LogicalTensor, ...]:
    return (
        tensors.audio_ids,
        tensors.audio_head_hidden,
        *omnivoice_weight_tensors(weights),
    )
