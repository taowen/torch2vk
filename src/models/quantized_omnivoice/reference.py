"""Generated PyTorch reference comparison functions."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from omnivoice.models.omnivoice import OmniVoice
from models.quantized_omnivoice.tensors.model import model_tensors
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.logical import ComparePolicy
from torch2vk.runtime.pytorch_debug import compare_expected
from torch2vk.runtime.session import RuntimeSession


ReferenceInput = np.ndarray | torch.Tensor
ReferenceExpected = dict[str, object]

_COMPARE_POLICIES = {
    "tensor": ComparePolicy(kind="tensor", rtol=1e-2, atol=1.5),
    "q4_tensor": ComparePolicy(kind="tensor", rtol=3e-2, atol=10.0),
    "token": ComparePolicy(kind="token"),
}


class ArrayReference(Protocol):
    def execute(self, inputs: dict[str, np.ndarray]) -> ReferenceExpected: ...


_MODEL: OmniVoice | None = None
_audio_head_reference: torch.nn.Module | None = None


def set_model(model: OmniVoice) -> None:
    global _MODEL
    _MODEL = model
    _clear_cached_references()


def _clear_cached_references() -> None:
    global _audio_head_reference
    _audio_head_reference = None


def _require_model() -> OmniVoice:
    if _MODEL is None:
        raise RuntimeError("reference.set_model(model) must be called before exported references are used")
    return _MODEL


def _load_audio_head() -> torch.nn.Module:
    global _audio_head_reference
    if _audio_head_reference is None:
        _audio_head_reference = _require_model().get_submodule('audio_heads')
        _audio_head_reference.eval()
    return _audio_head_reference

def _execute_and_compare(
    rt: RuntimeSession,
    *,
    name: str,
    reference: ArrayReference | torch.nn.Module,
    tensors: object,
    output_bindings: dict[str, str],
    policy: ComparePolicy | dict[str, ComparePolicy],
    inputs: dict[str, ReferenceInput],
) -> ReferenceExpected:
    expected = _execute_reference(
        reference=reference,
        inputs={key: as_numpy_array(value) for key, value in inputs.items()},
        output_bindings=output_bindings,
    )
    compare_expected(
        rt,
        name=name,
        tensors=tensors,
        output_bindings=output_bindings,
        expected=expected,
        policy=policy,
    )
    return expected


def _execute_reference(
    *,
    reference: ArrayReference | torch.nn.Module,
    inputs: dict[str, np.ndarray],
    output_bindings: dict[str, str],
) -> ReferenceExpected:
    if isinstance(reference, torch.nn.Module):
        args = [
            torch.from_numpy(np.ascontiguousarray(value)).cuda()
            for value in inputs.values()
        ]
        with torch.no_grad():
            output = reference(*args)
        if isinstance(output, tuple):
            if len(output) != 1:
                raise RuntimeError(f"reference module returned {len(output)} outputs")
            output = output[0]
        return {_single_output_name(output_bindings): output}
    return reference.execute(inputs)


def _single_output_name(output_bindings: dict[str, str]) -> str:
    if len(output_bindings) != 1:
        raise RuntimeError(
            f"module reference requires exactly one output binding, got {sorted(output_bindings)}"
        )
    return next(iter(output_bindings))


def _policy(policy: str | dict[str, str]) -> ComparePolicy | dict[str, ComparePolicy]:
    if isinstance(policy, dict):
        return {key: _COMPARE_POLICIES[value] for key, value in policy.items()}
    return _COMPARE_POLICIES[policy]

def run_input_embed(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    step: int,
    input_ids: ReferenceInput,
    audio_mask: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name=f'omnivoice.step.{step:04d}.input_embed',
        reference=reference,
        tensors=model_tensors(),
        output_bindings={'hidden_states': 'llm_forward.hidden_states'},
        policy=_policy('tensor'),
        inputs={
            "input_ids": input_ids,
            "audio_mask": audio_mask,
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
    return _execute_and_compare(
        rt,
        name=f'omnivoice.step.{step:04d}.token_score',
        reference=reference,
        tensors=model_tensors(),
        output_bindings={'candidate_scores': 'candidate_scores'},
        policy=_policy({'candidate_tokens': 'token', 'candidate_scores': 'tensor'}),
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
    return _execute_and_compare(
        rt,
        name=f'omnivoice.step.{step:04d}.token_update',
        reference=reference,
        tensors=model_tensors(),
        output_bindings={'tokens': 'tokens', 'batch_input_ids': 'batch_input_ids'},
        policy=_policy('token'),
        inputs={
            "tokens": tokens,
            "batch_input_ids": batch_input_ids,
            "candidate_tokens": candidate_tokens,
            "candidate_scores": candidate_scores,
            "unmask_count": unmask_count,
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
    return _execute_and_compare(
        rt,
        name=f'omnivoice.step.{step:04d}.llm_forward',
        reference=reference,
        tensors=model_tensors(),
        output_bindings={'mul_365': 'llm_forward.mul_365'},
        policy=_policy('q4_tensor'),
        inputs={
            "hidden_states": hidden_states,
            "cos": cos,
            "sin": sin,
            "attention_mask": attention_mask,
        },
    )

def run_audio_head(
    rt: RuntimeSession,
    *,
    step: int,
    input: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name=f'omnivoice.step.{step:04d}.audio_head',
        reference=_load_audio_head(),
        tensors=model_tensors(),
        output_bindings={'linear': 'audio_head.linear'},
        policy=_policy('tensor'),
        inputs={
            "input": input,
        },
    )
