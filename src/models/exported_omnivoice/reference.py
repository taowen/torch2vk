"""Generated PyTorch reference comparison functions."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from models.exported_omnivoice import reference_specs
from models.exported_omnivoice.tensors.model import model_tensors
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.logical import ComparePolicy
from torch2vk.runtime.pytorch_debug import compare_expected_with_spec
from torch2vk.runtime.reference import ReferenceSpec
from torch2vk.runtime.session import RuntimeSession


ReferenceInput = np.ndarray | torch.Tensor
ReferenceExpected = dict[str, object]

_COMPARE_POLICIES = {
    "tensor": ComparePolicy(kind="tensor", rtol=1e-2, atol=1.5),
    "token": ComparePolicy(kind="token"),
}


class ArrayReference(Protocol):
    def execute(self, inputs: dict[str, np.ndarray]) -> ReferenceExpected: ...


def _execute_and_compare(
    rt: RuntimeSession,
    *,
    name: str,
    reference: ArrayReference,
    tensors: object,
    spec: ReferenceSpec,
    inputs: dict[str, ReferenceInput],
) -> ReferenceExpected:
    expected = reference.execute(
        {key: as_numpy_array(value) for key, value in inputs.items()}
    )
    compare_expected_with_spec(
        rt,
        name=name,
        tensors=tensors,
        spec=spec,
        expected=expected,
        policy=_policy_for_spec(spec),
    )
    return expected


def _policy_for_spec(spec: ReferenceSpec) -> ComparePolicy | dict[str, ComparePolicy]:
    if isinstance(spec.policy, dict):
        return {key: _COMPARE_POLICIES[value] for key, value in spec.policy.items()}
    return _COMPARE_POLICIES[spec.policy]


def run_audio_head(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    step: int,
    input: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.AUDIO_HEAD_SPEC
    return _execute_and_compare(
        rt,
        name=f'omnivoice.step.{step:04d}.audio_head',
        reference=reference,
        tensors=model_tensors(),
        spec=spec,
        inputs={
            "input": input,
        },
    )

def run_input_embed(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    step: int,
    input_ids: ReferenceInput,
    audio_mask: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.INPUT_EMBED_SPEC
    return _execute_and_compare(
        rt,
        name=f'omnivoice.step.{step:04d}.input_embed',
        reference=reference,
        tensors=model_tensors(),
        spec=spec,
        inputs={
            "input_ids": input_ids,
            "audio_mask": audio_mask,
        },
    )

def run_llm_forward(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    step: int,
    hidden_states: ReferenceInput,
    cos: ReferenceInput,
    sin: ReferenceInput,
    attention_mask: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.LLM_FORWARD_SPEC
    return _execute_and_compare(
        rt,
        name=f'omnivoice.step.{step:04d}.llm_forward',
        reference=reference,
        tensors=model_tensors(),
        spec=spec,
        inputs={
            "hidden_states": hidden_states,
            "cos": cos,
            "sin": sin,
            "attention_mask": attention_mask,
        },
    )

def run_token_score(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    step: int,
    logits: ReferenceInput,
    tokens: ReferenceInput,
    audio_mask_id: ReferenceInput,
    rng_seed: ReferenceInput,
    step_index: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.TOKEN_SCORE_SPEC
    return _execute_and_compare(
        rt,
        name=f'omnivoice.step.{step:04d}.token_score',
        reference=reference,
        tensors=model_tensors(),
        spec=spec,
        inputs={
            "logits": logits,
            "tokens": tokens,
            "audio_mask_id": audio_mask_id,
            "rng_seed": rng_seed,
            "step_index": step_index,
        },
    )

def run_token_update(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    step: int,
    tokens: ReferenceInput,
    batch_input_ids: ReferenceInput,
    candidate_tokens: ReferenceInput,
    candidate_scores: ReferenceInput,
    unmask_count: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.TOKEN_UPDATE_SPEC
    return _execute_and_compare(
        rt,
        name=f'omnivoice.step.{step:04d}.token_update',
        reference=reference,
        tensors=model_tensors(),
        spec=spec,
        inputs={
            "tokens": tokens,
            "batch_input_ids": batch_input_ids,
            "candidate_tokens": candidate_tokens,
            "candidate_scores": candidate_scores,
            "unmask_count": unmask_count,
        },
    )
