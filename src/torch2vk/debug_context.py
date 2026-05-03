"""Eager Vulkan debug context with PyTorch reference comparison."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from .artifacts import bound_tensor_to_torch
from .logical import LogicalTensor
from .pytorch import ArtifactCache, TransformFn, ensure_pytorch_reference
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
    _reference_ready: bool = False

    def dispatch(
        self,
        variant: ShaderVariant,
        pytorch_model: object,
        tensors: Mapping[str, LogicalTensor],
    ) -> None:
        self.ensure_reference(pytorch_model)
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
        self._compare_written_tensors(record, tensors)

    def ensure_reference(self, pytorch_model: object) -> None:
        if self._reference_ready:
            return
        self.reference = ensure_pytorch_reference(
            model=pytorch_model,
            tensors=self.tensor_sequence,
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
        )

    def _compare_written_tensors(
        self,
        record: DispatchRecord,
        dispatch_tensors: Mapping[str, LogicalTensor],
    ) -> None:
        for field_name in record.writes:
            tensor = dispatch_tensors[field_name]
            if tensor.pytorch_probe is None or tensor.compare is None:
                continue
            self.candidate[tensor.name] = bound_tensor_to_torch(
                _first_lookup_tensor(self.tensors[tensor.name]),
                self.allocations,
            )
            difference = artifact_difference(
                tensor.name,
                reference=self.reference,
                candidate=self.candidate,
                policy=tensor.compare,
            )
            if difference is None:
                continue
            raise AssertionError(
                "\n".join(
                    (
                        f"first mismatch: {tensor.name}",
                        f"writer shader: {record.shader}",
                        f"writer dispatch: {record.index}",
                        f"reason: {difference.reason}",
                    )
                )
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
