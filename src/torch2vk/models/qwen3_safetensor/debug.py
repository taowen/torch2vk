"""Qwen3 safetensor debug input declarations."""

from __future__ import annotations

from torch2vk.logical import LogicalTensor

from .tensors.decode import Qwen3DecodeTensors
from .tensors.prefill import Qwen3PrefillTensors
from .tensors.weights import Qwen3Weights, qwen3_weight_tensors


def qwen3_prefill_initial_tensors(
    *,
    tensors: Qwen3PrefillTensors,
    weights: Qwen3Weights,
) -> tuple[LogicalTensor, ...]:
    return (
        tensors.input_ids,
        tensors.position_ids,
        tensors.row_indices,
        tensors.rope_freq_factors_placeholder,
        tensors.attention_mask,
        *(layer.attention_sinks_placeholder for layer in tensors.layers),
        *qwen3_weight_tensors(weights),
    )


def qwen3_decode_initial_tensors(
    *,
    tensors: Qwen3DecodeTensors,
    weights: Qwen3Weights,
) -> tuple[LogicalTensor, ...]:
    return (
        tensors.input_ids,
        tensors.position_ids,
        tensors.row_indices,
        tensors.rope_freq_factors_placeholder,
        tensors.attention_mask,
        *(layer.attention_sinks_placeholder for layer in tensors.layers),
        *qwen3_weight_tensors(weights),
    )
