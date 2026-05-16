"""Generated PyTorch reference comparison functions."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from models.quantized_qwen3.tensors.model import model_tensors
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.logical import ComparePolicy
from torch2vk.runtime.pytorch_debug import compare_expected
from torch2vk.runtime.session import RuntimeSession


ReferenceInput = np.ndarray | torch.Tensor | int
ReferenceExpected = dict[str, object]

_COMPARE_POLICIES = {
    "tensor": ComparePolicy(kind="tensor", rtol=1e-2, atol=1.5),
    "q8_tensor": ComparePolicy(kind="tensor", rtol=2e-2, atol=8.0),
    "q4_tensor": ComparePolicy(kind="tensor", rtol=5e-2, atol=128.0),
    "token": ComparePolicy(kind="token"),
}


class ArrayReference(Protocol):
    def execute(self, inputs: dict[str, np.ndarray]) -> ReferenceExpected: ...


_MODEL: Qwen3ForCausalLM | None = None


def set_model(model: Qwen3ForCausalLM) -> None:
    global _MODEL
    _MODEL = model
    _clear_cached_references()


def _clear_cached_references() -> None:
    pass


def _require_model() -> Qwen3ForCausalLM:
    if _MODEL is None:
        raise RuntimeError("reference.set_model(model) must be called before exported references are used")
    return _MODEL




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
        inputs={key: _reference_input_array(value) for key, value in inputs.items()},
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
        float_dtype = _module_float_dtype(reference)
        args = [
            _reference_arg(value, float_dtype)
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


def _reference_input_array(value: ReferenceInput) -> np.ndarray:
    if isinstance(value, int):
        return np.asarray([value], dtype=np.int64)
    return as_numpy_array(value)


def _reference_arg(value: np.ndarray, float_dtype: torch.dtype | None) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(value)).cuda()
    if float_dtype is not None and tensor.is_floating_point():
        return tensor.to(dtype=float_dtype)
    return tensor


def _module_float_dtype(module: torch.nn.Module) -> torch.dtype | None:
    for parameter in module.parameters(recurse=True):
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in module.buffers(recurse=True):
        if buffer.is_floating_point():
            return buffer.dtype
    return None


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

def run_embed_tokens(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    input: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name='embed_tokens',
        reference=reference,
        tensors=model_tensors().embed_tokens,
        output_bindings={'embedding': 'embedding'},
        policy=_policy('tensor'),
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
    past_key_values: ReferenceInput,
    attention_mask: ReferenceInput,
    key_cache: ReferenceInput,
    value_cache: ReferenceInput,
    cache_position: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name=f'qwen3.prefill.layer.{layer_idx}',
        reference=reference,
        tensors=model_tensors().text_layers[layer_idx],
        output_bindings={'add_7': 'add_7', 'index_copy': 'index_copy', 'index_copy_1': 'index_copy_1'},
        policy=_policy('tensor'),
        inputs={
            "hidden_states": hidden_states,
            "position_embeddings_0": position_embeddings_0,
            "position_embeddings_1": position_embeddings_1,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "cache_position": cache_position,
        },
    )

def run_text_norm(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    hidden_states: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name='text_norm',
        reference=reference,
        tensors=model_tensors().text_norm,
        output_bindings={'mul_1': 'mul_1'},
        policy=_policy('tensor'),
        inputs={
            "hidden_states": hidden_states,
        },
    )

def run_decode_embed(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    input: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name='decode_embed',
        reference=reference,
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
    past_key_values: ReferenceInput,
    attention_mask: ReferenceInput,
    key_cache: ReferenceInput,
    value_cache: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name=f'qwen3.decode.{step:04d}.layer.{layer_idx}',
        reference=reference,
        tensors=model_tensors().decode_layers[layer_idx],
        output_bindings={'add_7': 'add_7', 'index_copy': 'index_copy', 'index_copy_1': 'index_copy_1'},
        policy=_policy('tensor'),
        inputs={
            "hidden_states": hidden_states,
            "position_embeddings_0": position_embeddings_0,
            "position_embeddings_1": position_embeddings_1,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "key_cache": key_cache,
            "value_cache": value_cache,
        },
    )

def run_decode_norm(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    hidden_states: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name='decode_norm',
        reference=reference,
        tensors=model_tensors().decode_norm,
        output_bindings={'mul_1': 'mul_1'},
        policy=_policy('tensor'),
        inputs={
            "hidden_states": hidden_states,
        },
    )
