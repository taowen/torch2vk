"""Run a Vulkan stage inside a PyTorch-driven reference flow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias

import numpy as np
import torch

from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.host_array import prepare_host_array
from torch2vk.runtime.logical import ComparePolicy, LogicalTensor, MemoryClass, TensorRole
from torch2vk.runtime.pytorch_debug import compare_expected
from torch2vk.runtime.session import RuntimeSession
from torch2vk.vulkan.types import tensor_nbytes


ReferenceInput: TypeAlias = np.ndarray | torch.Tensor | int
ReferenceExpected: TypeAlias = dict[str, object]


def compare_vulkan_stage(
    rt: RuntimeSession,
    *,
    name: str,
    run: Callable[[], None],
    tensors: object,
    input_bindings: Mapping[str, str],
    output_bindings: dict[str, str],
    inputs: Mapping[str, ReferenceInput],
    expected: ReferenceExpected,
    policy: ComparePolicy | dict[str, ComparePolicy],
) -> ReferenceExpected:
    request_inputs: dict[str, object] = {}
    request_state: dict[LogicalTensor, object] = {}
    stage_inputs: dict[LogicalTensor, np.ndarray] = {}
    for key, field_name in input_bindings.items():
        tensor = _logical_tensor_path(tensors, field_name)
        value = _reference_input_array(inputs[key], tensor)
        if tensor.memory is MemoryClass.REQUEST_STATE:
            request_state[tensor] = value
        elif tensor.role is TensorRole.INPUT:
            request_inputs[tensor.name] = value
        else:
            stage_inputs[tensor] = value
    normalized_expected = _normalize_expected_outputs(
        tensors=tensors,
        output_bindings=output_bindings,
        expected=expected,
    )
    with rt.request(inputs=request_inputs, state=request_state):
        for tensor, value in stage_inputs.items():
            _initialize_stage_tensor(rt, tensor, value)
        with rt.frame(name):
            run()
            compare_expected(
                rt,
                name=name,
                tensors=tensors,
                output_bindings=output_bindings,
                expected=normalized_expected,
                policy=policy,
            )
    return normalized_expected


def _normalize_expected_outputs(
    *,
    tensors: object,
    output_bindings: Mapping[str, str],
    expected: ReferenceExpected,
) -> ReferenceExpected:
    normalized = dict(expected)
    for key, field_name in output_bindings.items():
        tensor = _logical_tensor_path(tensors, field_name)
        normalized[key] = _reference_input_array(expected[key], tensor)
    return normalized


def _reference_input_array(value: object, tensor: LogicalTensor) -> np.ndarray:
    if isinstance(value, int):
        array = np.asarray([value], dtype=np.int64)
    else:
        array = as_numpy_array(value)
    try:
        expected_dtype = np.dtype(tensor.spec.dtype)
    except TypeError:
        return array
    if array.dtype == expected_dtype:
        return array
    if np.issubdtype(expected_dtype, np.floating) and np.issubdtype(array.dtype, np.floating):
        return array.astype(expected_dtype)
    return array


def _initialize_stage_tensor(
    rt: RuntimeSession,
    tensor: LogicalTensor,
    value: np.ndarray,
) -> None:
    array = prepare_host_array(tensor, value, context="streaming compare input")
    expected = tensor_nbytes(tensor.spec)
    if array.nbytes != expected:
        raise ValueError(
            f"{tensor.name} streaming compare input has {array.nbytes} bytes, expected {expected}"
        )
    ((slice_, allocation),) = rt.device.upload_numpy_arrays_with_allocations(
        [(tensor.name, array)]
    )
    with tensor.runtime_write_scope():
        tensor.buffer = slice_
        tensor.descriptor_nbytes = expected
        tensor.alias_source = None
    rt._request_allocations.append(allocation)
    rt._request_tensors.add(tensor)


def _logical_tensor_path(tensors: object, field_path: str) -> LogicalTensor:
    value: object = tensors
    for segment in field_path.split("."):
        if isinstance(value, (tuple, list)) and segment.isdecimal():
            value = value[int(segment)]
        else:
            value = getattr(value, segment)
    if not isinstance(value, LogicalTensor):
        raise TypeError(f"{type(tensors).__name__}.{field_path} is not a LogicalTensor")
    return value
