"""Shader contracts and dispatch recording."""

from __future__ import annotations

import re
import struct
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
class ResourceBinding:
    name: str
    binding: int
    descriptor_type: str


@dataclass(frozen=True, slots=True)
class UniformBlock:
    name: str
    binding: int
    values: tuple[int | str, ...]


@dataclass(frozen=True, slots=True)
class PushConstantField:
    name: str
    offset: int
    dtype: str
    value: float | str


@dataclass(frozen=True, slots=True)
class PushConstantBlock:
    size: int
    fields: tuple[PushConstantField, ...] = ()

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"Push constant size must be positive, got {self.size}")


def _empty_specialization_constants() -> Mapping[int, int]:
    return {}


def _empty_symbol_constraints() -> Mapping[str, int]:
    return {}


@dataclass(frozen=True, slots=True)
class ShaderContract:
    name: str
    inputs: Mapping[str, TensorContract]
    outputs: Mapping[str, TensorContract]
    bindings: tuple[Binding, ...]
    dispatch: tuple[int | str, int | str, int | str]
    resources: tuple[ResourceBinding, ...] = ()
    uniforms: tuple[UniformBlock, ...] = ()
    push_constants: PushConstantBlock | None = None
    symbol_constraints: Mapping[str, int] = field(default_factory=_empty_symbol_constraints)

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
        for expression, expected_value in self.symbol_constraints.items():
            actual_value = _resolve_symbolic_int(self.name, expression, symbols)
            if actual_value != expected_value:
                raise ValueError(
                    f"{self.name} requires {expression} == {expected_value}, got {actual_value}"
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
    specialization_constants: Mapping[int, int] = field(
        default_factory=_empty_specialization_constants
    )

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
    uniforms: Mapping[str, tuple[int, ...]]
    push_constant_size: int | None
    push_constants: bytes | None


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
                uniforms=resolve_uniform_blocks(variant.contract, symbols),
                push_constant_size=None
                if variant.contract.push_constants is None
                else variant.contract.push_constants.size,
                push_constants=pack_push_constants(variant.contract, tensors, symbols),
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
    for binding in variant.contract.resources:
        if binding.binding not in source_bindings:
            raise ValueError(
                f"{variant.name}.{binding.name} resource binding {binding.binding} "
                "is missing from GLSL source"
            )
    for uniform in variant.contract.uniforms:
        if uniform.binding not in source_bindings:
            raise ValueError(
                f"{variant.name}.{uniform.name} uniform binding {uniform.binding} "
                "is missing from GLSL source"
            )
    if "push_constant" in variant.source and variant.contract.push_constants is None:
        raise ValueError(f"{variant.name} source declares push constants but contract has none")


def resolve_uniform_blocks(
    contract: ShaderContract,
    symbols: Mapping[str, int],
) -> dict[str, tuple[int, ...]]:
    return {
        uniform.name: tuple(
            _resolve_symbolic_int(contract.name, value, symbols) for value in uniform.values
        )
        for uniform in contract.uniforms
    }


def pack_uniform_blocks(
    contract: ShaderContract,
    symbols: Mapping[str, int],
) -> dict[str, bytes]:
    return {
        name: struct.pack(f"<{len(values)}i", *values)
        for name, values in resolve_uniform_blocks(contract, symbols).items()
    }


def dispatch_dimensions(
    contract: ShaderContract,
    symbols: Mapping[str, int],
) -> tuple[int, int, int]:
    x, y, z = (
        _resolve_symbolic_int(contract.name, value, symbols)
        for value in contract.dispatch
    )
    return x, y, z


def pack_push_constants(
    contract: ShaderContract,
    tensors: Mapping[str, LogicalTensor],
    symbols: Mapping[str, int],
) -> bytes | None:
    block = contract.push_constants
    if block is None:
        return None
    data = bytearray(block.size)
    for constant in block.fields:
        value = _resolve_push_constant_value(contract.name, constant.value, tensors, symbols)
        payload = _pack_push_constant_value(contract.name, constant, value)
        end = constant.offset + len(payload)
        if constant.offset < 0 or end > block.size:
            raise ValueError(
                f"{contract.name}.{constant.name} push constant range "
                f"[{constant.offset}, {end}) exceeds {block.size}"
            )
        data[constant.offset:end] = payload
    return bytes(data)


def _shader_source_bindings(source: str) -> set[int]:
    return {
        int(match)
        for match in re.findall(
            r"layout\s*\([^)]*\bbinding\s*=\s*(\d+)\b[^)]*\)",
            source,
        )
    }


def _resolve_symbolic_int(
    contract_name: str,
    value: int | str,
    symbols: Mapping[str, int],
) -> int:
    if isinstance(value, int):
        return value
    if "*" in value:
        product = 1
        for part in value.split("*"):
            product *= _resolve_symbolic_int(contract_name, part.strip(), symbols)
        return product
    if value.isdecimal():
        return int(value)
    resolved = symbols.get(value)
    if resolved is None:
        raise ValueError(f"{contract_name} uniform references unresolved symbol {value!r}")
    return resolved


def _resolve_push_constant_value(
    contract_name: str,
    value: float | str,
    tensors: Mapping[str, LogicalTensor],
    symbols: Mapping[str, int],
) -> int | float:
    if isinstance(value, int | float):
        return value
    if value.isdecimal():
        return int(value)
    if "*" in value:
        product = 1
        for part in value.split("*"):
            resolved = _resolve_push_constant_value(
                contract_name,
                part.strip(),
                tensors,
                symbols,
            )
            product *= int(resolved)
        return product
    if value in symbols:
        return symbols[value]
    if value.endswith(".numel"):
        return _tensor_numel(contract_name, tensors, value.removesuffix(".numel"))
    if ".dim" in value:
        field, _, dim_text = value.partition(".dim")
        return _tensor_dim(contract_name, tensors, field, int(dim_text))
    raise ValueError(f"{contract_name} push constant references unresolved value {value!r}")


def _pack_push_constant_value(
    contract_name: str,
    field: PushConstantField,
    value: float,
) -> bytes:
    if field.dtype == "uint32":
        return struct.pack("<I", int(value))
    if field.dtype == "int32":
        return struct.pack("<i", int(value))
    if field.dtype == "float32":
        return struct.pack("<f", float(value))
    raise ValueError(
        f"{contract_name}.{field.name} unsupported push constant dtype {field.dtype!r}"
    )


def _tensor_numel(
    contract_name: str,
    tensors: Mapping[str, LogicalTensor],
    field: str,
) -> int:
    elements = 1
    for dim in _tensor_shape(contract_name, tensors, field):
        elements *= dim
    return elements


def _tensor_dim(
    contract_name: str,
    tensors: Mapping[str, LogicalTensor],
    field: str,
    dim: int,
) -> int:
    shape = _tensor_shape(contract_name, tensors, field)
    if dim < 0 or dim >= len(shape):
        raise ValueError(f"{contract_name}.{field} has no dimension {dim}")
    return shape[dim]


def _tensor_shape(
    contract_name: str,
    tensors: Mapping[str, LogicalTensor],
    field: str,
) -> tuple[int, ...]:
    tensor = tensors.get(field)
    if tensor is None:
        raise ValueError(f"{contract_name} push constant references unknown field {field!r}")
    shape: list[int] = []
    for dim in tensor.shape:
        if not isinstance(dim, int):
            raise TypeError(f"{contract_name}.{field} has symbolic shape {tensor.shape}")
        shape.append(dim)
    return tuple(shape)


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
        _reserve_descriptor_binding(contract.name, binding.binding, seen_bindings)
        if binding.field not in fields:
            raise ValueError(f"{contract.name} binding references unknown field {binding.field!r}")
        if binding.access is BindingAccess.WRITE and binding.field not in contract.outputs:
            raise ValueError(f"{contract.name}.{binding.field} write binding must be an output")
    for resource in contract.resources:
        _reserve_descriptor_binding(contract.name, resource.binding, seen_bindings)
        if not resource.name:
            raise ValueError(f"{contract.name} resource binding must have a name")
    for uniform in contract.uniforms:
        _reserve_descriptor_binding(contract.name, uniform.binding, seen_bindings)
        if not uniform.name:
            raise ValueError(f"{contract.name} uniform binding must have a name")
        if not uniform.values:
            raise ValueError(f"{contract.name}.{uniform.name} uniform must declare values")


def _reserve_descriptor_binding(
    contract_name: str,
    binding: int,
    seen_bindings: set[int],
) -> None:
    if binding in seen_bindings:
        raise ValueError(f"{contract_name} duplicate descriptor binding {binding}")
    seen_bindings.add(binding)


def _validate_dispatch(
    shader_name: str,
    dispatch: Sequence[int | str],
    symbols: Mapping[str, int],
) -> None:
    if len(dispatch) != 3:
        raise ValueError(f"{shader_name} dispatch must have 3 dimensions")
    for dim in dispatch:
        value = _resolve_symbolic_int(shader_name, dim, symbols) if isinstance(dim, str) else dim
        if value <= 0:
            raise ValueError(f"{shader_name} dispatch dimension must be positive, got {dim!r}")
