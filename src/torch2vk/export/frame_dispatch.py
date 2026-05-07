"""Runtime op dispatch for models that need dynamic dispatch (e.g., alias ops)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

from torch2vk.export.lowering import DEFAULT_LOWERING_REGISTRY, OpLoweringRegistry
from torch2vk.export.protocols import ExportOpLike
from torch2vk.runtime.shader import IOKind, ShaderVariant

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


def dispatch_op(
    rt: "RuntimeSession",
    op: ExportOpLike,
    ns: Any,
    shader_variants: Mapping[str, ShaderVariant],
    *,
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
) -> None:
    """Dispatch a single op to its resolved shader using getattr on ns."""
    binding = lowering_registry.resolve(op=op)
    if binding is None:
        raise NotImplementedError(
            f"No shader binding for op target={op.target!r} inputs={op.inputs!r}"
        )
    shader = shader_variants.get(binding.shader)
    if shader is None:
        raise KeyError(f"Shader variant {binding.shader!r} not found")

    input_fields = [f for f in shader.contract.fields if f.io_kind is IOKind.INPUT]
    output_fields = [
        f for f in shader.contract.fields if f.io_kind in (IOKind.OUTPUT, IOKind.INOUT)
    ]

    kwargs: dict[str, Any] = {}
    for i, field in enumerate(input_fields):
        if i < len(op.inputs):
            kwargs[field.name] = getattr(ns, op.inputs[i]) if hasattr(ns, op.inputs[i]) else ns[op.inputs[i]]
    for j, field in enumerate(output_fields):
        if j < len(op.outputs):
            kwargs[field.name] = getattr(ns, op.outputs[j]) if hasattr(ns, op.outputs[j]) else ns[op.outputs[j]]

    shader(rt, **kwargs)
