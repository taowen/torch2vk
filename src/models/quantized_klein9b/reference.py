"""Generated PyTorch reference comparison functions."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from models.quantized_klein9b.pytorch_modules import Flux2
from models.quantized_klein9b.tensors.model import model_tensors
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


_MODEL: Flux2 | None = None


def set_model(model: Flux2) -> None:
    global _MODEL
    _MODEL = model
    _clear_cached_references()


def _clear_cached_references() -> None:
    pass


def _require_model() -> Flux2:
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

def run_flux(
    rt: RuntimeSession,
    reference: ArrayReference,
    *,
    x: ReferenceInput,
    x_ids: ReferenceInput,
    timesteps: ReferenceInput,
    ctx: ReferenceInput,
    ctx_ids: ReferenceInput,
    guidance: ReferenceInput,
) -> ReferenceExpected:
    return _execute_and_compare(
        rt,
        name='klein9b.flux',
        reference=reference,
        tensors=model_tensors().flux,
        output_bindings={'linear_120': 'linear_120'},
        policy=_policy('q4_tensor'),
        inputs={
            "x": x,
            "x_ids": x_ids,
            "timesteps": timesteps,
            "ctx": ctx,
            "ctx_ids": ctx_ids,
            "guidance": guidance,
        },
    )
