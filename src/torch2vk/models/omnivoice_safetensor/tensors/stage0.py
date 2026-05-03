"""OmniVoice stage0 LogicalTensor tree."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.logical import (
    ComparePolicy,
    LogicalTensor,
    PyTorchProbe,
    TensorRole,
    activation_tensor,
    input_tensor,
    output_tensor,
)
from torch2vk.models.omnivoice_safetensor.spec import OmniVoiceSpec


@dataclass(frozen=True, slots=True)
class OmniVoiceStage0Tensors:
    audio_ids: LogicalTensor
    audio_embedding: LogicalTensor
    audio_embedding_sum: LogicalTensor
    audio_head_hidden: LogicalTensor
    audio_head_logits: LogicalTensor
    audio_head_logits_rounded: LogicalTensor
    argmax_ids: LogicalTensor
    argmax_scores: LogicalTensor
    updated_ids: LogicalTensor
    selected_flat_index: LogicalTensor
    selected_score: LogicalTensor
    selected_candidate_id: LogicalTensor


def omnivoice_stage0_tensors(
    *,
    batch: int,
    steps: int,
    spec: OmniVoiceSpec,
) -> OmniVoiceStage0Tensors:
    codebooks = spec.num_audio_codebook
    vocab = spec.audio_vocab_size * spec.num_audio_codebook
    hidden = spec.qwen3.hidden_size
    return OmniVoiceStage0Tensors(
        audio_ids=input_tensor(
            "stage0.audio_ids",
            dtype="int32",
            shape=(batch, codebooks, steps),
        ),
        audio_embedding=activation_tensor(
            "stage0.audio_embedding",
            dtype="float32",
            shape=(batch, codebooks, steps, hidden),
        ),
        audio_embedding_sum=activation_tensor(
            "stage0.audio_embedding_sum",
            dtype="float32",
            shape=(batch, steps, hidden),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage0.audio_embedding.output",
            ),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        audio_head_hidden=activation_tensor(
            "stage0.audio_head.hidden",
            dtype="float32",
            shape=(batch, steps, hidden),
        ),
        audio_head_logits=activation_tensor(
            "stage0.audio_head.logits",
            dtype="float32",
            shape=(batch, steps, vocab),
            role=TensorRole.LOGITS,
            pytorch_probe=PyTorchProbe(kind="manual", source="stage0.audio_head.logits"),
            compare=ComparePolicy(kind="tensor", rtol=1e-4, atol=1e-4),
        ),
        audio_head_logits_rounded=activation_tensor(
            "stage0.audio_head.logits_rounded",
            dtype="float32",
            shape=(batch, steps, vocab),
            role=TensorRole.LOGITS,
        ),
        argmax_ids=output_tensor(
            "stage0.audio_head.argmax_ids",
            dtype="int32",
            shape=(batch, codebooks, steps),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="stage0.audio_head.tokens",
                normalize="int32",
            ),
            compare=ComparePolicy(kind="token"),
        ),
        argmax_scores=output_tensor(
            "stage0.audio_head.argmax_scores",
            dtype="float32",
            shape=(batch, codebooks, steps),
        ),
        updated_ids=output_tensor(
            "stage0.updated_ids",
            dtype="int32",
            shape=(batch, codebooks, steps),
        ),
        selected_flat_index=output_tensor(
            "stage0.selected_flat_index",
            dtype="int32",
            shape=(batch,),
        ),
        selected_score=output_tensor(
            "stage0.selected_score",
            dtype="float32",
            shape=(batch,),
        ),
        selected_candidate_id=output_tensor(
            "stage0.selected_candidate_id",
            dtype="int32",
            shape=(batch,),
        ),
    )
