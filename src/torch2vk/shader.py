"""Shader contracts and dispatch recording."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .logical import ROW_MAJOR_LAYOUT, LogicalTensor, TensorLayout


class BindingAccess(StrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@dataclass(frozen=True, slots=True)
class TensorContract:
    dtype: str
    shape: tuple[int | str, ...]
    layout: TensorLayout = ROW_MAJOR_LAYOUT


@dataclass(frozen=True, slots=True)
class Binding:
    field: str
    binding: int
    access: BindingAccess
    descriptor_type: str = "storage_buffer"


@dataclass(frozen=True, slots=True)
class ShaderContract:
    name: str
    inputs: Mapping[str, TensorContract]
    outputs: Mapping[str, TensorContract]
    bindings: tuple[Binding, ...]
    dispatch: tuple[int | str, int | str, int | str]

    def validate(self, tensors: Mapping[str, LogicalTensor]) -> dict[str, int]:
        expected = set(self.inputs) | set(self.outputs)
        missing = expected - set(tensors)
        if missing:
            raise ValueError(f"{self.name} missing tensor fields: {sorted(missing)}")
        unknown = set(tensors) - expected
        if unknown:
            raise ValueError(f"{self.name} got unknown tensor fields: {sorted(unknown)}")

        symbols: dict[str, int] = {}
        for field_name, contract in {**self.inputs, **self.outputs}.items():
            tensor = tensors[field_name]
            _validate_tensor_contract(
                shader_name=self.name,
                field=field_name,
                contract=contract,
                tensor=tensor,
                symbols=symbols,
            )
        _validate_bindings(self)
        _validate_dispatch(self.name, self.dispatch, symbols)
        return symbols

    @property
    def read_fields(self) -> tuple[str, ...]:
        return tuple(
            binding.field
            for binding in self.bindings
            if binding.access in (BindingAccess.READ, BindingAccess.READ_WRITE)
        )

    @property
    def write_fields(self) -> tuple[str, ...]:
        return tuple(
            binding.field
            for binding in self.bindings
            if binding.access in (BindingAccess.WRITE, BindingAccess.READ_WRITE)
        )


@dataclass(frozen=True, slots=True)
class ShaderVariant:
    name: str
    family: str
    contract: ShaderContract
    source: str = ""
    compile_defines: tuple[str, ...] = ()
    include_dirs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ShaderVariant name must be non-empty")
        if self.contract.name != self.name:
            raise ValueError(
                f"ShaderVariant {self.name!r} contract name must match, got {self.contract.name!r}"
            )

    def __call__(self, target: DispatchTarget, **tensors: LogicalTensor) -> None:
        target.run(self, **tensors)


@dataclass(frozen=True, slots=True)
class DispatchRecord:
    index: int
    shader: str
    family: str
    reads: Mapping[str, str]
    writes: Mapping[str, str]
    symbols: Mapping[str, int]


def _new_dispatch_records() -> list[DispatchRecord]:
    return []


@dataclass(slots=True)
class DispatchTarget:
    records: list[DispatchRecord] = field(default_factory=_new_dispatch_records)
    validate: bool = True

    def run(self, variant: ShaderVariant, **tensors: LogicalTensor) -> None:
        symbols = variant.contract.validate(tensors) if self.validate else {}
        self.records.append(
            DispatchRecord(
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
            )
        )

    def reset(self) -> None:
        self.records.clear()


def validate_shader_source_bindings(variant: ShaderVariant) -> None:
    source_bindings = _shader_source_bindings(variant.source)
    if not source_bindings:
        raise ValueError(f"{variant.name} source has no GLSL layout(binding=...) declarations")
    for binding in variant.contract.bindings:
        if binding.binding not in source_bindings:
            raise ValueError(
                f"{variant.name}.{binding.field} contract binding {binding.binding} "
                "is missing from GLSL source"
            )


def _shader_source_bindings(source: str) -> set[int]:
    return {
        int(match)
        for match in re.findall(
            r"layout\s*\([^)]*\bbinding\s*=\s*(\d+)\b[^)]*\)",
            source,
        )
    }


def _validate_tensor_contract(
    *,
    shader_name: str,
    field: str,
    contract: TensorContract,
    tensor: LogicalTensor,
    symbols: dict[str, int],
) -> None:
    if tensor.dtype != contract.dtype:
        raise ValueError(
            f"{shader_name}.{field} expected dtype {contract.dtype}, got {tensor.dtype} "
            f"for {tensor.name}"
        )
    if len(tensor.shape) != len(contract.shape):
        raise ValueError(
            f"{shader_name}.{field} expected rank {len(contract.shape)}, got {len(tensor.shape)} "
            f"for {tensor.name}"
        )
    if tensor.layout != contract.layout:
        raise ValueError(
            f"{shader_name}.{field} expected layout {contract.layout}, got {tensor.layout} "
            f"for {tensor.name}"
        )
    for index, (actual, expected) in enumerate(zip(tensor.shape, contract.shape, strict=True)):
        if isinstance(expected, int):
            if actual != expected:
                raise ValueError(
                    f"{shader_name}.{field}[{index}] expected {expected}, "
                    f"got {actual} for {tensor.name}"
                )
            continue
        if not isinstance(actual, int):
            raise TypeError(
                f"{shader_name}.{field}[{index}] cannot resolve symbol {expected!r} "
                f"from symbolic actual {actual!r} for {tensor.name}"
            )
        previous = symbols.setdefault(expected, actual)
        if previous != actual:
            raise ValueError(
                f"{shader_name}.{field}[{index}] symbol {expected!r} expected {previous}, "
                f"got {actual} for {tensor.name}"
            )


def _validate_bindings(contract: ShaderContract) -> None:
    seen_bindings: set[int] = set()
    fields = set(contract.inputs) | set(contract.outputs)
    for binding in contract.bindings:
        if binding.binding in seen_bindings:
            raise ValueError(f"{contract.name} duplicate descriptor binding {binding.binding}")
        seen_bindings.add(binding.binding)
        if binding.field not in fields:
            raise ValueError(f"{contract.name} binding references unknown field {binding.field!r}")
        if binding.access is BindingAccess.WRITE and binding.field not in contract.outputs:
            raise ValueError(f"{contract.name}.{binding.field} write binding must be an output")


def _validate_dispatch(
    shader_name: str,
    dispatch: Sequence[int | str],
    symbols: Mapping[str, int],
) -> None:
    if len(dispatch) != 3:
        raise ValueError(f"{shader_name} dispatch must have 3 dimensions")
    for dim in dispatch:
        value = symbols[dim] if isinstance(dim, str) else dim
        if value <= 0:
            raise ValueError(f"{shader_name} dispatch dimension must be positive, got {dim!r}")
