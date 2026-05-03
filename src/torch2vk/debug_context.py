"""Eager Vulkan debug context with record-first reference comparison."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from .artifacts import bound_tensor_to_torch
from .logical import LogicalTensor
from .pytorch import ArtifactCache, ReferenceProvider, TransformFn
from .shader import DispatchRecord, ShaderVariant, pack_push_constants, resolve_uniform_blocks
from .validation import artifact_difference
from .vulkan_backend import VulkanBuffer
from .vulkan_runner import (
    LogicalTensorLookup,
    VulkanDescriptorBuffer,
    VulkanShaderDispatch,
    storage_descriptor_buffers,
)


def _empty_records() -> list[DispatchRecord]:
    return []


def _empty_artifacts() -> dict[str, torch.Tensor]:
    return {}


def _empty_scope_stack() -> list[str]:
    return []


@dataclass(slots=True)
class DebugContext:
    shader_dir: Path
    variants: Mapping[str, ShaderVariant]
    context: Any
    tensors: LogicalTensorLookup
    tensor_sequence: tuple[LogicalTensor, ...]
    allocations: Mapping[str, VulkanBuffer]
    inputs: Mapping[str, Any]
    cache: ArtifactCache
    resource_buffers: Mapping[str, Mapping[str, VulkanDescriptorBuffer]] | None = None
    transforms: Mapping[str, TransformFn] | None = None
    extra_fingerprint: Mapping[str, Any] | None = None
    records: list[DispatchRecord] = field(default_factory=_empty_records)
    reference: dict[str, torch.Tensor] = field(default_factory=_empty_artifacts)
    candidate: dict[str, torch.Tensor] = field(default_factory=_empty_artifacts)
    _scope_stack: list[str] = field(default_factory=_empty_scope_stack)
    _reference_ready: bool = False

    @contextmanager
    def scope(self, name: str, **labels: int | str) -> Iterator[None]:
        parts = [name, *(f"{key}={value}" for key, value in sorted(labels.items()))]
        self._scope_stack.append("/".join(parts))
        try:
            yield
        finally:
            self._scope_stack.pop()

    def dispatch(
        self,
        variant: ShaderVariant,
        tensors: Mapping[str, LogicalTensor],
    ) -> None:
        record = self._record(variant, tensors)
        dispatch = VulkanShaderDispatch.load(self.context, variant, shader_dir=self.shader_dir)
        resources: Mapping[str, Mapping[str, VulkanDescriptorBuffer]] = (
            {} if self.resource_buffers is None else self.resource_buffers
        )
        try:
            dispatch.run(
                tensors=tensors,
                tensor_buffers=storage_descriptor_buffers(
                    _iter_lookup_tensors(self.tensors),
                    allocations=self.allocations,
                ),
                resource_buffers=resources.get(record.shader),
            )
        finally:
            dispatch.close()
        self.records.append(record)
        self._readback_written_tensors(record, tensors)

    def ensure_reference(
        self,
        reference_provider: ReferenceProvider,
        tensors: tuple[LogicalTensor, ...] | None = None,
    ) -> None:
        if self._reference_ready:
            return
        self.reference = reference_provider.ensure(
            tensors=self.tensor_sequence if tensors is None else tensors,
            inputs=self.inputs,
            cache=self.cache,
            transforms=self.transforms,
            extra_fingerprint=self.extra_fingerprint,
        )
        self._reference_ready = True

    def _record(
        self,
        variant: ShaderVariant,
        tensors: Mapping[str, LogicalTensor],
    ) -> DispatchRecord:
        symbols = variant.contract.validate(tensors)
        return DispatchRecord(
            index=len(self.records),
            shader=variant.name,
            family=variant.family,
            reads={
                field: tensors[field].name
                for field in variant.contract.read_fields
                if field in tensors
            },
            writes={
                field: tensors[field].name
                for field in variant.contract.write_fields
                if field in tensors
            },
            symbols=symbols,
            uniforms=resolve_uniform_blocks(variant.contract, symbols),
            push_constant_size=None
            if variant.contract.push_constants is None
            else variant.contract.push_constants.size,
            push_constants=pack_push_constants(variant.contract, tensors, symbols),
            scope=self.current_scope,
        )

    @property
    def current_scope(self) -> str:
        return ".".join(self._scope_stack)

    def compare_records(self, reference_provider: ReferenceProvider) -> None:
        required = self.comparable_written_tensors()
        if not required:
            return
        self.ensure_reference(reference_provider, tensors=required)
        for record in self.records:
            for tensor_name in record.writes.values():
                tensor = _first_lookup_tensor(self.tensors[tensor_name])
                if tensor.pytorch_probe is None or tensor.compare is None:
                    continue
                artifact_key = _artifact_key(record, tensor.name, self.reference, self.candidate)
                difference = artifact_difference(
                    artifact_key,
                    reference=self.reference,
                    candidate=self.candidate,
                    policy=tensor.compare,
                )
                if difference is None:
                    continue
                raise AssertionError(
                    "\n".join(
                        (
                            f"first mismatch: {artifact_key}",
                            f"writer shader: {record.shader}",
                            f"writer dispatch: {record.index}",
                            f"reason: {difference.reason}",
                        )
                    )
                )

    def comparable_written_tensors(self) -> tuple[LogicalTensor, ...]:
        found: list[LogicalTensor] = []
        seen: set[str] = set()
        for record in self.records:
            for tensor_name in record.writes.values():
                if tensor_name in seen:
                    continue
                tensor = _first_lookup_tensor(self.tensors[tensor_name])
                if tensor.pytorch_probe is None or tensor.compare is None:
                    continue
                seen.add(tensor.name)
                found.append(tensor)
        return tuple(found)

    def _readback_written_tensors(
        self,
        record: DispatchRecord,
        dispatch_tensors: Mapping[str, LogicalTensor],
    ) -> None:
        for field_name in record.writes:
            tensor = dispatch_tensors[field_name]
            if tensor.pytorch_probe is None or tensor.compare is None:
                continue
            self.candidate[record.artifact_key(tensor.name)] = bound_tensor_to_torch(
                _first_lookup_tensor(self.tensors[tensor.name]),
                self.allocations,
            )


def _iter_lookup_tensors(tensors: LogicalTensorLookup) -> tuple[LogicalTensor, ...]:
    values: list[LogicalTensor] = []
    for value in tensors.values():
        if isinstance(value, LogicalTensor):
            values.append(value)
        else:
            values.extend(value)
    return tuple(values)


def _first_lookup_tensor(value: LogicalTensor | tuple[LogicalTensor, ...]) -> LogicalTensor:
    return value if isinstance(value, LogicalTensor) else value[0]


def _artifact_key(
    record: DispatchRecord,
    tensor_name: str,
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> str:
    scoped = record.artifact_key(tensor_name)
    if scoped in reference or scoped in candidate:
        return scoped
    return tensor_name
