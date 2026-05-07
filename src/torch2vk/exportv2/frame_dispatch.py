"""Runtime op dispatch for models that need dynamic dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from torch2vk.exportv2.lowering import DEFAULT_LOWERING_REGISTRY, OpLoweringRegistry
from torch2vk.exportv2.protocols import ExportOpLike
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.shader import IOKind, ShaderVariant

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


def dispatch_op(
    rt: "RuntimeSession",
    op: ExportOpLike,
    env: Mapping[str, LogicalTensor],
    shader_variants: Mapping[str, ShaderVariant],
    *,
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
) -> None:
    """Dispatch a single op to its resolved shader using logical tensors from env."""
    target = op.target
    inputs = tuple(op.inputs)
    outputs = tuple(op.outputs)
    binding = lowering_registry.resolve_target_inputs(target=target, inputs=inputs)
    if binding is None:
        raise NotImplementedError(f"No shader binding for op target={target!r} inputs={inputs!r}")
    shader = shader_variants.get(binding.shader)
    if shader is None:
        raise KeyError(f"Shader variant {binding.shader!r} not found")

    input_fields = [field for field in shader.contract.fields if field.io_kind is IOKind.INPUT]
    output_fields = [
        field for field in shader.contract.fields if field.io_kind in (IOKind.OUTPUT, IOKind.INOUT)
    ]

    kwargs: dict[str, LogicalTensor] = {}
    for index, field in enumerate(input_fields):
        if index < len(inputs):
            kwargs[field.name] = env[inputs[index]]
    for index, field in enumerate(output_fields):
        if index < len(outputs):
            kwargs[field.name] = env[outputs[index]]

    shader(rt, **kwargs)
