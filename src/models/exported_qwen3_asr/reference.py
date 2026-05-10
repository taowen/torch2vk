"""Generated PyTorch reference comparison functions."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)
from models.exported_qwen3_asr.tensors.model import model_tensors
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.logical import ComparePolicy
from torch2vk.runtime.pytorch_debug import compare_expected
from torch2vk.runtime.reference import ExportedProgramReference, load_exported_reference
from torch2vk.runtime.session import RuntimeSession


ReferenceInput = np.ndarray | torch.Tensor
ReferenceExpected = dict[str, object]

_COMPARE_POLICIES = {
    "tensor": ComparePolicy(kind="tensor", rtol=1e-2, atol=1.5),
    "token": ComparePolicy(kind="token"),
}


class ArrayReference(Protocol):
    def execute(self, inputs: dict[str, np.ndarray]) -> ReferenceExpected: ...


_MODEL: Qwen3ASRForConditionalGeneration | None = None
_embed_tokens_reference: ExportedProgramReference | None = None
_text_norm_reference: ExportedProgramReference | None = None
_lm_head_reference: ExportedProgramReference | None = None
_decode_embed_reference: ExportedProgramReference | None = None
_decode_norm_reference: ExportedProgramReference | None = None
_decode_lm_head_reference: ExportedProgramReference | None = None


def set_model(model: Qwen3ASRForConditionalGeneration) -> None:
    global _MODEL
    _MODEL = model
    _clear_cached_references()


def _clear_cached_references() -> None:
    global _embed_tokens_reference
    _embed_tokens_reference = None
    global _text_norm_reference
    _text_norm_reference = None
    global _lm_head_reference
    _lm_head_reference = None
    global _decode_embed_reference
    _decode_embed_reference = None
    global _decode_norm_reference
    _decode_norm_reference = None
    global _decode_lm_head_reference
    _decode_lm_head_reference = None


def _require_model() -> Qwen3ASRForConditionalGeneration:
    if _MODEL is None:
        raise RuntimeError("reference.set_model(model) must be called before exported references are used")
    return _MODEL


def _load_embed_tokens() -> ExportedProgramReference:
    global _embed_tokens_reference
    if _embed_tokens_reference is None:
        _embed_tokens_reference = load_exported_reference(
            Path(__file__).parent,
            'reference_programs/embed_tokens.pt2',
            state_dict=_require_model().get_submodule('thinker.model.embed_tokens').state_dict(),
        )
    return _embed_tokens_reference

def _load_text_norm() -> ExportedProgramReference:
    global _text_norm_reference
    if _text_norm_reference is None:
        _text_norm_reference = load_exported_reference(
            Path(__file__).parent,
            'reference_programs/text_norm.pt2',
            state_dict=_require_model().get_submodule('thinker.model.norm').state_dict(),
        )
    return _text_norm_reference

def _load_lm_head() -> ExportedProgramReference:
    global _lm_head_reference
    if _lm_head_reference is None:
        _lm_head_reference = load_exported_reference(
            Path(__file__).parent,
            'reference_programs/lm_head.pt2',
            state_dict=_require_model().get_submodule('thinker.lm_head').state_dict(),
        )
    return _lm_head_reference

def _load_decode_embed() -> ExportedProgramReference:
    global _decode_embed_reference
    if _decode_embed_reference is None:
        _decode_embed_reference = load_exported_reference(
            Path(__file__).parent,
            'reference_programs/decode_embed.pt2',
            state_dict=_require_model().get_submodule('thinker.model.embed_tokens').state_dict(),
        )
    return _decode_embed_reference

def _load_decode_norm() -> ExportedProgramReference:
    global _decode_norm_reference
    if _decode_norm_reference is None:
        _decode_norm_reference = load_exported_reference(
            Path(__file__).parent,
            'reference_programs/decode_norm.pt2',
            state_dict=_require_model().get_submodule('thinker.model.norm').state_dict(),
        )
    return _decode_norm_reference

def _load_decode_lm_head() -> ExportedProgramReference:
    global _decode_lm_head_reference
    if _decode_lm_head_reference is None:
        _decode_lm_head_reference = load_exported_reference(
            Path(__file__).parent,
            'reference_programs/decode_lm_head.pt2',
            state_dict=_require_model().get_submodule('thinker.lm_head').state_dict(),
        )
    return _decode_lm_head_reference

def _execute_and_compare(
    rt: RuntimeSession,
    *,
    name: str,
    reference: ArrayReference,
    tensors: object,
    output_bindings: dict[str, str],
    policy: ComparePolicy | dict[str, ComparePolicy],
    inputs: dict[str, ReferenceInput],
) -> ReferenceExpected:
    expected = reference.execute(
        {key: as_numpy_array(value) for key, value in inputs.items()}
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


def _policy(policy: str | dict[str, str]) -> ComparePolicy | dict[str, ComparePolicy]:
    if isinstance(policy, dict):
        return {key: _COMPARE_POLICIES[value] for key, value in policy.items()}
    return _COMPARE_POLICIES[policy]

def run_token_select(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    name: str,
    logits: ReferenceInput,
    eos_token_ids: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name=f'{name}',
        reference=reference,
        tensors=model_tensors(),
        output_bindings={'next_token': 'next_token', 'done': 'done'},
        policy=_policy('token'),
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
    return _execute_and_compare(
        rt,
        name=f'{name}',
        reference=reference,
        tensors=model_tensors(),
        output_bindings={'generated_tokens': 'generated_tokens', 'generated_length': 'generated_length', 'stopped': 'stopped'},
        policy=_policy('token'),
        inputs={
            "next_token": next_token,
            "token_index": token_index,
            "done": done,
        },
    )

def run_audio_encoder(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    x: ReferenceInput,
    position_embedding: ReferenceInput,
    compact_index: ReferenceInput,
    attention_mask: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name='spike.audio.encoder',
        reference=reference,
        tensors=model_tensors().audio_encoder,
        output_bindings={'linear_110': 'linear_110'},
        policy=_policy('tensor'),
        inputs={
            "x": x,
            "position_embedding": position_embedding,
            "compact_index": compact_index,
            "attention_mask": attention_mask,
        },
    )

def run_embed_tokens(
    rt: RuntimeSession,
    *,
    input: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name='spike.text.embed',
        reference=_load_embed_tokens(),
        tensors=model_tensors().embed_tokens,
        output_bindings={'embedding': 'embedding'},
        policy=_policy('tensor'),
        inputs={
            "input": input,
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
    return _execute_and_compare(
        rt,
        name='spike.text.audio_inject',
        reference=reference,
        tensors=model_tensors().audio_inject,
        output_bindings={'embedding': 'index_copy'},
        policy=_policy('tensor'),
        inputs={
            "inputs_embeds": inputs_embeds,
            "audio_positions": audio_positions,
            "audio_features": audio_features,
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
    return _execute_and_compare(
        rt,
        name=f'spike.text.layer.{layer_idx}',
        reference=reference,
        tensors=model_tensors().text_layers[layer_idx],
        output_bindings={'add_7': 'add_7'},
        policy=_policy('tensor'),
        inputs={
            "hidden_states": hidden_states,
            "position_embeddings_0": position_embeddings_0,
            "position_embeddings_1": position_embeddings_1,
            "cache_position": cache_position,
        },
    )

def run_text_norm(
    rt: RuntimeSession,
    *,
    hidden_states: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name='spike.text.norm',
        reference=_load_text_norm(),
        tensors=model_tensors().text_norm,
        output_bindings={'mul_1': 'mul_1'},
        policy=_policy('tensor'),
        inputs={
            "hidden_states": hidden_states,
        },
    )

def run_lm_head(
    rt: RuntimeSession,
    *,
    input: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name='spike.text.lm_head',
        reference=_load_lm_head(),
        tensors=model_tensors().lm_head,
        output_bindings={'linear': 'linear'},
        policy=_policy('tensor'),
        inputs={
            "input": input,
        },
    )

def run_decode_embed(
    rt: RuntimeSession,
    *,
    step: int,
    input: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name=f'spike.decode.{step:04d}.embed',
        reference=_load_decode_embed(),
        tensors=model_tensors().decode_embed,
        output_bindings={'embedding': 'embedding'},
        policy=_policy('tensor'),
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
    return _execute_and_compare(
        rt,
        name=f'spike.decode.{step:04d}.layer.{layer_idx}',
        reference=reference,
        tensors=model_tensors().decode_layers[layer_idx],
        output_bindings={'add_7': 'add_7'},
        policy=_policy('tensor'),
        inputs={
            "hidden_states": hidden_states,
            "position_embeddings_0": position_embeddings_0,
            "position_embeddings_1": position_embeddings_1,
            "cache_position": cache_position,
        },
    )

def run_decode_norm(
    rt: RuntimeSession,
    *,
    step: int,
    hidden_states: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name=f'spike.decode.{step:04d}.norm',
        reference=_load_decode_norm(),
        tensors=model_tensors().decode_norm,
        output_bindings={'mul_1': 'mul_1'},
        policy=_policy('tensor'),
        inputs={
            "hidden_states": hidden_states,
        },
    )

def run_decode_lm_head(
    rt: RuntimeSession,
    *,
    step: int,
    input: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name=f'spike.decode.{step:04d}.lm_head',
        reference=_load_decode_lm_head(),
        tensors=model_tensors().decode_lm_head,
        output_bindings={'linear': 'linear'},
        policy=_policy('tensor'),
        inputs={
            "input": input,
        },
    )
