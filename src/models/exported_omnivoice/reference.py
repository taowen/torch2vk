"""Generated PyTorch reference comparison functions."""

from __future__ import annotations

import numpy as np
import torch

from models.exported_omnivoice.run import _run_input_embed as _dispatch_input_embed
from models.exported_omnivoice.dispatch.llm_forward import run_llm_forward as _dispatch_llm_forward
from models.exported_omnivoice.dispatch.audio_head import run_audio_head as _dispatch_audio_head
from models.exported_omnivoice.dispatch.audio_decode import run_audio_decode as _dispatch_audio_decode
from models.exported_omnivoice.tensors.model import model_tensors
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.logical import ComparePolicy
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.streaming_compare import compare_vulkan_stage


ReferenceInput = np.ndarray | torch.Tensor | int
ReferenceExpected = dict[str, object]

_COMPARE_POLICIES = {
    "tensor": ComparePolicy(kind="tensor", rtol=1e-2, atol=1.5),
    "q8_tensor": ComparePolicy(kind="tensor", rtol=2e-2, atol=8.0),
    "q4_tensor": ComparePolicy(kind="tensor", rtol=5e-2, atol=128.0),
    "token": ComparePolicy(kind="token"),
}


def _reference_int(value: ReferenceInput) -> int:
    if isinstance(value, int):
        return value
    return int(as_numpy_array(value).reshape(-1)[0])


def _policy(policy: str | dict[str, str]) -> ComparePolicy | dict[str, ComparePolicy]:
    if isinstance(policy, dict):
        return {key: _COMPARE_POLICIES[value] for key, value in policy.items()}
    return _COMPARE_POLICIES[policy]

def compare_input_embed(
    rt: RuntimeSession,
    *,
    step: int,
    expected: ReferenceExpected,
    input_ids: ReferenceInput,
    audio_mask: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name=f'omnivoice.step.{step:04d}.input_embed',
        run=lambda: _dispatch_input_embed(rt),
        tensors=model_tensors(),
        input_bindings={'input_ids': 'batch_input_ids', 'audio_mask': 'batch_audio_mask'},
        output_bindings={'hidden_states': 'llm_forward.hidden_states'},
        policy=_policy('tensor'),
        inputs={
            "input_ids": input_ids,
            "audio_mask": audio_mask,
        },
        expected=expected,
    )

def compare_llm_forward(
    rt: RuntimeSession,
    *,
    step: int,
    expected: ReferenceExpected,
    hidden_states: ReferenceInput,
    cos: ReferenceInput,
    sin: ReferenceInput,
    attention_mask: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name=f'omnivoice.step.{step:04d}.llm_forward',
        run=lambda: _dispatch_llm_forward(rt),
        tensors=model_tensors(),
        input_bindings={'hidden_states': 'llm_forward.hidden_states', 'cos': 'rope.cos', 'sin': 'rope.sin', 'attention_mask': 'attention_mask'},
        output_bindings={'mul_365': 'llm_forward.mul_365'},
        policy=_policy('tensor'),
        inputs={
            "hidden_states": hidden_states,
            "cos": cos,
            "sin": sin,
            "attention_mask": attention_mask,
        },
        expected=expected,
    )

def compare_audio_head(
    rt: RuntimeSession,
    *,
    step: int,
    expected: ReferenceExpected,
    input: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name=f'omnivoice.step.{step:04d}.audio_head',
        run=lambda: _dispatch_audio_head(rt),
        tensors=model_tensors(),
        input_bindings={'input': 'audio_head.input'},
        output_bindings={'linear': 'audio_head.linear'},
        policy=_policy('tensor'),
        inputs={
            "input": input,
        },
        expected=expected,
    )

def compare_audio_decode(
    rt: RuntimeSession,
    *,
    expected: ReferenceExpected,
    audio_codes: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name='omnivoice.audio_decode',
        run=lambda: _dispatch_audio_decode(rt),
        tensors=model_tensors(),
        input_bindings={'audio_codes': 'tokens'},
        output_bindings={'conv1d_31': 'audio_decode.conv1d_31'},
        policy=_policy('tensor'),
        inputs={
            "audio_codes": audio_codes,
        },
        expected=expected,
    )
