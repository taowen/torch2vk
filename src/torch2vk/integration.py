"""Shared Vulkan/PyTorch integration debug harness."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from .debug_context import DebugContext
from .logical import LogicalTensor
from .pytorch import ArtifactCache, PyTorchModelReferenceProvider, ReferenceProvider, TransformFn
from .shader import DispatchRecord, ShaderVariant
from .storage import bind_storage, plan_storage
from .validation import validate_dispatch_read_write_chain
from .vulkan_backend import VulkanBuffer, VulkanContext, create_compute_context
from .vulkan_runner import (
    LogicalTensorLookup,
    VulkanDescriptorBuffer,
    allocate_storage_buffers,
    write_bound_tensor_payloads,
)

type InitialTensorWriter = Callable[
    [LogicalTensorLookup, Mapping[str, VulkanBuffer], tuple[LogicalTensor, ...]],
    None,
]
type ResourceFactory = Callable[[VulkanContext], Mapping[str, Mapping[str, VulkanDescriptorBuffer]]]
type ResourceCloser = Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class DebugIntegrationCase:
    shader_dir: Path
    shader_package: str
    allocation_id: str
    tensors: object
    weights: object
    initial_tensors: tuple[LogicalTensor, ...]
    inputs: Mapping[str, Any]
    reference_provider: ReferenceProvider
    run: Callable[[DebugContext], None]
    write_initial_tensors: InitialTensorWriter | None = None
    weight_payloads: Mapping[str, bytes] | None = None
    resource_factory: ResourceFactory | None = None
    resource_closer: ResourceCloser | None = None
    transforms: Mapping[str, TransformFn] | None = None
    extra_fingerprint: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DebugIntegrationResult:
    records: tuple[DispatchRecord, ...]


def run_debug_integration_case(case: DebugIntegrationCase) -> DebugIntegrationResult:
    unbound = (
        *collect_logical_tensors(case.tensors),
        *collect_logical_tensors(case.weights),
    )
    plan = plan_storage(unbound, allocation_id=case.allocation_id)
    bound = bind_storage(unbound, plan)
    tensor_lookup = logical_tensor_lookup(bound)
    variants = shader_variants(case.shader_package)

    context = create_compute_context()
    try:
        with ExitStack() as stack:
            allocations = allocate_storage_buffers(context, plan)
            for allocation in allocations.values():
                stack.callback(allocation.close)

            resources = _create_resources(case.resource_factory, context)
            if case.resource_closer is not None:
                stack.callback(case.resource_closer, resources)

            if case.write_initial_tensors is not None:
                case.write_initial_tensors(tensor_lookup, allocations, case.initial_tensors)
            if case.weight_payloads is not None:
                write_bound_tensor_payloads(tensor_lookup, allocations, case.weight_payloads)

            cache_dir = stack.enter_context(TemporaryDirectory())
            debug_context = DebugContext(
                shader_dir=case.shader_dir,
                variants=variants,
                context=context,
                tensors=tensor_lookup,
                tensor_sequence=bound,
                allocations=allocations,
                inputs=case.inputs,
                cache=ArtifactCache(Path(cache_dir)),
                resource_buffers=resources,
                transforms=case.transforms,
                extra_fingerprint=case.extra_fingerprint,
            )
            case.run(debug_context)
            debug_context.compare_records(case.reference_provider)
            validate_dispatch_read_write_chain(
                debug_context.records,
                initial_tensors=case.initial_tensors,
            ).raise_for_issues()
            return DebugIntegrationResult(records=tuple(debug_context.records))
    finally:
        context.close()


def collect_logical_tensors(value: object) -> tuple[LogicalTensor, ...]:
    found: list[LogicalTensor] = []
    _collect(value, found)
    return tuple(found)


def logical_tensor_lookup(
    tensors: Sequence[LogicalTensor],
) -> dict[str, LogicalTensor | tuple[LogicalTensor, ...]]:
    lookup: dict[str, list[LogicalTensor]] = {}
    for tensor in tensors:
        lookup.setdefault(tensor.name, []).append(tensor)
    return {
        name: values[0] if len(values) == 1 else tuple(values)
        for name, values in lookup.items()
    }


def first_tensor(value: LogicalTensor | tuple[LogicalTensor, ...]) -> LogicalTensor:
    return value if isinstance(value, LogicalTensor) else value[0]


def shader_variants(package_name: str) -> dict[str, ShaderVariant]:
    package = importlib.import_module(package_name)
    variants: dict[str, ShaderVariant] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, ShaderVariant):
                variants[value.name] = value
    return variants


def _collect(value: object, found: list[LogicalTensor]) -> None:
    if isinstance(value, LogicalTensor):
        found.append(value)
        return
    if isinstance(value, tuple):
        for item in cast("tuple[object, ...]", value):
            _collect(item, found)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect(getattr(value, field.name), found)


def _create_resources(
    factory: ResourceFactory | None,
    context: VulkanContext,
) -> Mapping[str, Mapping[str, VulkanDescriptorBuffer]]:
    if factory is None:
        return {}
    return factory(context)


def pytorch_model_reference_provider(model: Any) -> PyTorchModelReferenceProvider:
    return PyTorchModelReferenceProvider(model)
