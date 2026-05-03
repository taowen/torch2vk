"""Readable Qwen3 safetensor execution skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.logical import LogicalTensor, activation_tensor, input_tensor, output_tensor
from torch2vk.shader import DispatchTarget

from .schema import qwen3_weight_tensors
from .shaders import EMBEDDING_LOOKUP_BF16_F32, LINEAR_BF16_F32, RMS_NORM_F32
from .spec import Qwen3Spec


@dataclass(frozen=True, slots=True)
class Qwen3ExecutionTensors:
    input_ids: LogicalTensor
    hidden: LogicalTensor
    final_norm: LogicalTensor
    logits: LogicalTensor


def qwen3_execution_tensors(*, batch: int, steps: int, spec: Qwen3Spec) -> Qwen3ExecutionTensors:
    return Qwen3ExecutionTensors(
        input_ids=input_tensor("input.input_ids", dtype="int32", shape=(batch, steps)),
        hidden=activation_tensor(
            "decode.embedding",
            dtype="float32",
            shape=(batch, steps, spec.hidden_size),
        ),
        final_norm=activation_tensor(
            "decode.final_norm",
            dtype="float32",
            shape=(batch, steps, spec.hidden_size),
        ),
        logits=output_tensor(
            "output.logits",
            dtype="float32",
            shape=(batch, steps, spec.vocab_size),
        ),
    )


def record_qwen3_minimal_prefill(
    target: DispatchTarget,
    *,
    spec: Qwen3Spec,
    tensors: Qwen3ExecutionTensors,
) -> None:
    """Record a minimal top-level Qwen3 path.

    This is intentionally not a complete transformer yet. It establishes the
    source-visible execution shape and validates the initial contracts:
    embedding -> final norm -> lm head.
    """

    weights = {tensor.name: tensor for tensor in qwen3_weight_tensors(spec)}
    EMBEDDING_LOOKUP_BF16_F32(
        target,
        input_ids=tensors.input_ids,
        weight=weights["weights.embed_tokens"],
        output=tensors.hidden,
    )
    RMS_NORM_F32(
        target,
        x=tensors.hidden,
        weight=weights["weights.norm"],
        output=tensors.final_norm,
    )
    LINEAR_BF16_F32(
        target,
        x=tensors.final_norm,
        weight=weights["weights.lm_head"],
        output=tensors.logits,
    )
