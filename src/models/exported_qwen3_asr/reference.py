"""Generated PyTorch reference comparison functions."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from models.exported_qwen3_asr import reference_specs
from models.exported_qwen3_asr.tensors.model import model_tensors
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


def run_audio_encoder(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    x: ReferenceInput,
    position_embedding: ReferenceInput,
    compact_index: ReferenceInput,
    attention_mask: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.AUDIO_ENCODER_SPEC
    return _execute_and_compare(
        rt,
        name='spike.audio.encoder',
        reference=reference,
        tensors=model_tensors().audio_encoder,
        spec=spec,
        inputs={
            "x": x,
            "position_embedding": position_embedding,
            "compact_index": compact_index,
            "attention_mask": attention_mask,
        },
    )

def run_audio_inject(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    inputs_embeds: ReferenceInput,
    audio_positions: ReferenceInput,
    audio_features: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.AUDIO_INJECT_SPEC
    return _execute_and_compare(
        rt,
        name='spike.text.audio_inject',
        reference=reference,
        tensors=model_tensors().audio_inject,
        spec=spec,
        inputs={
            "inputs_embeds": inputs_embeds,
            "audio_positions": audio_positions,
            "audio_features": audio_features,
        },
    )

def run_decode_embed(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    step: int,
    input: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.DECODE_EMBED_SPEC
    return _execute_and_compare(
        rt,
        name=f'spike.decode.{step:04d}.embed',
        reference=reference,
        tensors=model_tensors().decode_embed,
        spec=spec,
        inputs={
            "input": input,
        },
    )

def run_decode_layer(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    step: int,
    layer_idx: int,
    hidden_states: ReferenceInput,
    position_embeddings_0: ReferenceInput,
    position_embeddings_1: ReferenceInput,
    cache_position: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.DECODE_LAYER_SPEC
    return _execute_and_compare(
        rt,
        name=f'spike.decode.{step:04d}.layer.{layer_idx}',
        reference=reference,
        tensors=model_tensors().decode_layers[layer_idx],
        spec=spec,
        inputs={
            "hidden_states": hidden_states,
            "position_embeddings_0": position_embeddings_0,
            "position_embeddings_1": position_embeddings_1,
            "cache_position": cache_position,
        },
    )

def run_decode_lm_head(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    step: int,
    input: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.DECODE_LM_HEAD_SPEC
    return _execute_and_compare(
        rt,
        name=f'spike.decode.{step:04d}.lm_head',
        reference=reference,
        tensors=model_tensors().decode_lm_head,
        spec=spec,
        inputs={
            "input": input,
        },
    )

def run_decode_norm(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    step: int,
    hidden_states: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.DECODE_NORM_SPEC
    return _execute_and_compare(
        rt,
        name=f'spike.decode.{step:04d}.norm',
        reference=reference,
        tensors=model_tensors().decode_norm,
        spec=spec,
        inputs={
            "hidden_states": hidden_states,
        },
    )

def run_embed_tokens(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    input: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.EMBED_TOKENS_SPEC
    return _execute_and_compare(
        rt,
        name='spike.text.embed',
        reference=reference,
        tensors=model_tensors().embed_tokens,
        spec=spec,
        inputs={
            "input": input,
        },
    )

def run_lm_head(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    input: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.LM_HEAD_SPEC
    return _execute_and_compare(
        rt,
        name='spike.text.lm_head',
        reference=reference,
        tensors=model_tensors().lm_head,
        spec=spec,
        inputs={
            "input": input,
        },
    )

def run_text_layer(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    layer_idx: int,
    hidden_states: ReferenceInput,
    position_embeddings_0: ReferenceInput,
    position_embeddings_1: ReferenceInput,
    cache_position: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.TEXT_LAYER_SPEC
    return _execute_and_compare(
        rt,
        name=f'spike.text.layer.{layer_idx}',
        reference=reference,
        tensors=model_tensors().text_layers[layer_idx],
        spec=spec,
        inputs={
            "hidden_states": hidden_states,
            "position_embeddings_0": position_embeddings_0,
            "position_embeddings_1": position_embeddings_1,
            "cache_position": cache_position,
        },
    )

def run_text_norm(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    hidden_states: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.TEXT_NORM_SPEC
    return _execute_and_compare(
        rt,
        name='spike.text.norm',
        reference=reference,
        tensors=model_tensors().text_norm,
        spec=spec,
        inputs={
            "hidden_states": hidden_states,
        },
    )

def run_token_select(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    name: str,
    logits: ReferenceInput,
    eos_token_ids: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.TOKEN_SELECT_SPEC
    return _execute_and_compare(
        rt,
        name=f'{name}',
        reference=reference,
        tensors=model_tensors(),
        spec=spec,
        inputs={
            "logits": logits,
            "eos_token_ids": eos_token_ids,
        },
    )

def run_token_store(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    name: str,
    next_token: ReferenceInput,
    token_index: ReferenceInput,
    done: ReferenceInput,
) -> ReferenceExpected:
    spec = reference_specs.TOKEN_STORE_SPEC
    return _execute_and_compare(
        rt,
        name=f'{name}',
        reference=reference,
        tensors=model_tensors(),
        spec=spec,
        inputs={
            "next_token": next_token,
            "token_index": token_index,
            "done": done,
        },
    )
