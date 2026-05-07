"""Generate tensor scaffold declarations from exportv2 nodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch2vk.exportv2.lowering import DEFAULT_LOWERING_REGISTRY, OpLoweringRegistry
from torch2vk.exportv2.fx import StaticNode
from torch2vk.exportv2.tensor_pattern import TensorFieldPattern
from torch2vk.runtime.shader import DTypeReference, IOKind, ShaderVariant, TensorFieldSpec


@dataclass(frozen=True, slots=True)
class TensorDataclassFieldDecl:
    name: str
    annotation: str = "LogicalTensor"


@dataclass(frozen=True, slots=True)
class TensorDataclassDecl:
    class_name: str
    fields: tuple[TensorDataclassFieldDecl, ...]


def logical_tensor_dataclass_from_patterns(
    *,
    class_name: str,
    fields: Sequence[TensorFieldPattern],
    annotation_overrides: Mapping[str, str] | None = None,
) -> TensorDataclassDecl:
    overrides = {} if annotation_overrides is None else dict(annotation_overrides)
    return TensorDataclassDecl(
        class_name=class_name,
        fields=tuple(
            TensorDataclassFieldDecl(
                name=field.field,
                annotation=overrides.get(field.field, "LogicalTensor"),
            )
            for field in fields
        ),
    )


def tensor_scaffold_fields_from_static_nodes(
    *,
    nodes: Sequence[StaticNode],
    shader_variants: Mapping[str, ShaderVariant],
    parameter_sources: Mapping[str, str] | None = None,
    extra_fields: Sequence[TensorFieldPattern] = (),
    external_fields: Sequence[str] = (),
    role_overrides: Mapping[str, str] | None = None,
    dtype_overrides: Mapping[str, str] | None = None,
    lowering_registry: OpLoweringRegistry = DEFAULT_LOWERING_REGISTRY,
    include_unresolved_ops: bool = True,
) -> tuple[TensorFieldPattern, ...]:
    """Infer tensor fields from nodes and their shader tensor contracts."""
    sources = {} if parameter_sources is None else dict(parameter_sources)
    role_overrides = {} if role_overrides is None else dict(role_overrides)
    dtype_overrides = {} if dtype_overrides is None else dict(dtype_overrides)
    external = set(external_fields)
    fields: dict[str, TensorFieldPattern] = {}

    def merge(
        name: str,
        *,
        spec: TensorFieldSpec | None,
        is_output: bool,
    ) -> None:
        if name in external:
            return
        previous = fields.get(name)
        source_parameter = sources.get(name)
        role = _field_role(
            spec=spec,
            is_output=is_output,
            source_parameter=source_parameter,
            previous=previous,
        )
        dtype = _field_dtype(spec)
        if name in role_overrides:
            role = role_overrides[name]
        if name in dtype_overrides:
            dtype = dtype_overrides[name]
        fields[name] = _merge_pattern(
            previous,
            TensorFieldPattern(
                field=name,
                source_parameter=source_parameter,
                dtype=dtype,
                role=role,
            ),
        )

    for target, inputs, outputs in nodes:
        binding = lowering_registry.resolve_target_inputs(target=target, inputs=inputs)
        if binding is None:
            if include_unresolved_ops:
                for name in inputs:
                    merge(name, spec=None, is_output=False)
                for name in outputs:
                    merge(name, spec=None, is_output=True)
            continue
        try:
            shader = shader_variants[binding.shader]
        except KeyError as exc:
            raise KeyError(f"{target} lowers to unknown shader {binding.shader!r}") from exc
        input_specs = tuple(
            field for field in shader.contract.fields if field.io_kind is IOKind.INPUT
        )
        output_specs = tuple(
            field
            for field in shader.contract.fields
            if field.io_kind in (IOKind.OUTPUT, IOKind.INOUT)
        )
        for name, spec in _bind_contract_fields(inputs, input_specs):
            merge(name, spec=spec, is_output=False)
        for name, spec in _bind_contract_fields(outputs, output_specs):
            merge(name, spec=spec, is_output=True)

    for field in extra_fields:
        if field.field in external:
            continue
        fields[field.field] = _merge_pattern(fields.get(field.field), field)

    return tuple(fields.values())


def render_parameter_fields_constant(
    constant_name: str,
    fields: Sequence[TensorFieldPattern],
) -> str:
    values = tuple(field for field in fields if field.source_parameter)
    lines = [f"{constant_name} = {{"]
    lines.extend(f"    {field.field!r}: {field.source_parameter!r}," for field in values)
    lines.append("}")
    return "\n".join(lines)


def render_tensor_dtype_constant(
    constant_name: str,
    fields: Sequence[TensorFieldPattern],
) -> str:
    values = tuple(field for field in fields if field.dtype)
    lines = [f"{constant_name}: dict[str, str] = {{"]
    lines.extend(f"    {field.field!r}: {field.dtype!r}," for field in values)
    lines.append("}")
    return "\n".join(lines)


def render_tensor_dataclass(declaration: TensorDataclassDecl) -> str:
    lines = [
        "@dataclass",
        f"class {declaration.class_name}:",
    ]
    if declaration.fields:
        lines.extend(f"    {field.name}: {field.annotation}" for field in declaration.fields)
    else:
        lines.append("    pass")
    return "\n".join(lines)


def render_tensor_dataclasses(declarations: Sequence[TensorDataclassDecl]) -> str:
    return "\n\n\n".join(render_tensor_dataclass(declaration) for declaration in declarations)


def _bind_contract_fields(
    names: Sequence[str],
    specs: Sequence[TensorFieldSpec],
) -> tuple[tuple[str, TensorFieldSpec | None], ...]:
    remaining = list(specs)
    bindings: list[tuple[str, TensorFieldSpec | None]] = []
    for name in names:
        exact_index = next(
            (index for index, spec in enumerate(remaining) if spec.name == name),
            None,
        )
        if exact_index is not None:
            bindings.append((name, remaining.pop(exact_index)))
        elif remaining:
            bindings.append((name, remaining.pop(0)))
        else:
            bindings.append((name, None))
    return tuple(bindings)


def _field_dtype(spec: TensorFieldSpec | None) -> str | None:
    if spec is None:
        return None
    dtype = spec.contract.dtype
    if isinstance(dtype, str):
        return dtype
    if isinstance(dtype, tuple) and len(dtype) == 1:
        return dtype[0]
    if isinstance(dtype, DTypeReference):
        return None
    return None


def _field_role(
    *,
    spec: TensorFieldSpec | None,
    is_output: bool,
    source_parameter: str | None,
    previous: TensorFieldPattern | None,
) -> str | None:
    if source_parameter is not None:
        return "weight"
    if spec is None:
        return "activation" if is_output else None
    contract_role = spec.role
    if spec.io_kind is IOKind.INOUT:
        return "state"
    if contract_role in {"weight", "bias"}:
        return "weight"
    if contract_role in {"state", "scratch"}:
        return contract_role
    if is_output:
        return "activation"
    if previous is not None and previous.role not in {None, "input"}:
        return previous.role
    return "input"


def _merge_pattern(
    previous: TensorFieldPattern | None,
    current: TensorFieldPattern,
) -> TensorFieldPattern:
    if previous is None:
        return current
    return TensorFieldPattern(
        field=previous.field,
        source_parameter=_merge_optional(
            previous.source_parameter,
            current.source_parameter,
            field=previous.field,
            label="source_parameter",
        ),
        note=previous.note or current.note,
        dtype=_merge_optional(
            previous.dtype,
            current.dtype,
            field=previous.field,
            label="dtype",
        ),
        role=_merge_role(previous.role, current.role),
    )


def _merge_optional(
    previous: str | None,
    current: str | None,
    *,
    field: str,
    label: str,
) -> str | None:
    if previous is None:
        return current
    if current is None or current == previous:
        return previous
    raise ValueError(f"{field} has conflicting {label}: {previous!r} vs {current!r}")


def _merge_role(previous: str | None, current: str | None) -> str | None:
    if previous is None:
        return current
    if current is None or current == previous:
        return previous
    if current == "input":
        return previous
    if previous == "input":
        return current
    if previous == "activation" and current == "state":
        return current
    raise ValueError(f"conflicting tensor roles: {previous!r} vs {current!r}")
