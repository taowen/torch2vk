"""Shader contract and variant declarations consumed by RuntimeSession."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field as dataclass_field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from torch2vk.vulkan.types import CONTIGUOUS_LAYOUT, Dim, TensorLayout
from torch2vk.vulkan.shader_execution_requirements import ShaderExecutionRequirements

if TYPE_CHECKING:
    from torch2vk.runtime.logical import LogicalTensor

@dataclass(frozen=True, slots=True)
class CeilDivExpr:
    lhs: ExprDim
    rhs: ExprDim


@dataclass(frozen=True, slots=True)
class MulExpr:
    lhs: ExprDim
    rhs: ExprDim


@dataclass(frozen=True, slots=True)
class AddExpr:
    lhs: ExprDim
    rhs: ExprDim


ExprDim: TypeAlias = Dim | CeilDivExpr | MulExpr | AddExpr


def ceil_div(lhs: ExprDim, rhs: ExprDim) -> CeilDivExpr:
    return CeilDivExpr(lhs=lhs, rhs=rhs)


def mul(lhs: ExprDim, rhs: ExprDim) -> MulExpr:
    return MulExpr(lhs=lhs, rhs=rhs)


def add(lhs: ExprDim, rhs: ExprDim) -> AddExpr:
    return AddExpr(lhs=lhs, rhs=rhs)


class DescriptorType(StrEnum):
    STORAGE_BUFFER = "storage_buffer"
    UNIFORM_BUFFER = "uniform_buffer"


class IOKind(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class PushConstantType(StrEnum):
    UINT32 = "uint32"
    INT32 = "int32"
    FLOAT32 = "float32"
    UINT64 = "uint64"


@dataclass(frozen=True, slots=True)
class DTypeReference:
    field_name: str


def same_as(field_name: str) -> DTypeReference:
    return DTypeReference(field_name=field_name)


@dataclass(frozen=True, slots=True)
class TensorContract:
    dtype: str | tuple[str, ...] | DTypeReference
    shape: tuple[ExprDim, ...]
    layout: TensorLayout = CONTIGUOUS_LAYOUT


@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    name: str
    io_kind: IOKind
    role: str
    contract: TensorContract
    descriptor_type: DescriptorType = DescriptorType.STORAGE_BUFFER


@dataclass(frozen=True, slots=True)
class PushConstantInput:
    name: str


PushConstantResolver: TypeAlias = Callable[[Mapping[str, object], Mapping[str, int]], int | float]
PushConstantValue: TypeAlias = ExprDim | PushConstantInput | PushConstantResolver | int | float


@dataclass(frozen=True, slots=True)
class PushConstantFieldSpec:
    name: str
    dtype: PushConstantType
    offset: int
    value: PushConstantValue
    dynamic: bool = False

    @property
    def size(self) -> int:
        return 8 if self.dtype is PushConstantType.UINT64 else 4


@dataclass(frozen=True, slots=True)
class PushConstantSpec:
    size: int
    fields: tuple[PushConstantFieldSpec, ...]


@dataclass(frozen=True, slots=True)
class ParamsBufferFieldSpec:
    name: str
    dtype: PushConstantType
    offset: int
    value: PushConstantValue

    @property
    def size(self) -> int:
        return 8 if self.dtype is PushConstantType.UINT64 else 4


@dataclass(frozen=True, slots=True)
class ParamsBufferSpec:
    size: int
    fields: tuple[ParamsBufferFieldSpec, ...]
    binding_index: int


@dataclass(frozen=True, slots=True)
class ShaderContract:
    class_name: str
    shader_name: str
    fields: tuple[TensorFieldSpec, ...]
    dispatch: tuple[ExprDim, ExprDim, ExprDim]
    push_constants: PushConstantSpec | None = None
    params_buffer: ParamsBufferSpec | None = None

    @property
    def input_fields(self) -> tuple[TensorFieldSpec, ...]:
        return tuple(field for field in self.fields if field.io_kind in (IOKind.INPUT, IOKind.INOUT))

    @property
    def output_fields(self) -> tuple[TensorFieldSpec, ...]:
        return tuple(field for field in self.fields if field.io_kind in (IOKind.OUTPUT, IOKind.INOUT))

    def validate_definition(self) -> None:
        if not self.class_name:
            raise ValueError("ShaderContract class_name must be non-empty")
        if not self.shader_name:
            raise ValueError("ShaderContract shader_name must be non-empty")
        if len(self.dispatch) != 3:
            raise ValueError(f"{self.shader_name} dispatch must be rank 3")
        names: set[str] = set()
        for field_spec in self.fields:
            if not field_spec.name:
                raise ValueError(f"{self.shader_name} has an empty field name")
            if field_spec.name in names:
                raise ValueError(f"{self.shader_name} duplicate field name {field_spec.name}")
            names.add(field_spec.name)
            if field_spec.descriptor_type is not DescriptorType.STORAGE_BUFFER:
                raise ValueError(f"{self.shader_name}.{field_spec.name} must be a storage buffer")
        if self.push_constants is not None:
            _validate_push_constant_definition(self.shader_name, self.push_constants)


@dataclass(frozen=True, slots=True)
class ShaderVariant:
    name: str
    family: str
    contract: ShaderContract
    source: str
    precompiled_spv_path: Path | None = None
    specialization_constants: tuple[tuple[int, int], ...] | None = None
    include_dirs: tuple[Path, ...] = ()
    compile_defines: tuple[str, ...] = ()
    execution_requirements: ShaderExecutionRequirements | None = None

    def __post_init__(self) -> None:
        if self.name != self.contract.shader_name:
            raise ValueError(f"ShaderVariant name {self.name!r} must equal contract shader_name")
        if not self.family:
            raise ValueError(f"{self.name} family must be non-empty")
        if not self.source and self.precompiled_spv_path is None:
            raise ValueError(f"{self.name} requires inline source or precompiled SPIR-V")
        self.contract.validate_definition()

    def __call__(self, rt: object, **arguments: object) -> None:
        dispatch = getattr(rt, "dispatch")
        dispatch(self, **arguments)


def collect_shader_variants(source: object) -> dict[str, "ShaderVariant"]:
    variants: dict[str, ShaderVariant] = {}
    for attr_name in dir(source):
        value = getattr(source, attr_name)
        if isinstance(value, ShaderVariant):
            variants[value.name] = value
    return variants


@dataclass(frozen=True, slots=True)
class DispatchTensorSnapshot:
    field: str
    tensor: str
    shape: tuple[int, ...]
    dtype: str
    descriptor_offset: int
    descriptor_nbytes: int
    version: int


@dataclass(frozen=True, slots=True)
class DispatchRecord:
    index: int
    frame: str
    shader: str
    reads: tuple[tuple[str, "LogicalTensor"], ...]
    writes: tuple[tuple[str, "LogicalTensor"], ...]
    logical_reads: tuple[tuple[str, str], ...]
    logical_writes: tuple[tuple[str, str], ...]
    symbols: tuple[tuple[str, int], ...]
    dispatch_size: tuple[int, int, int]
    descriptor_views: tuple[tuple[str, int, int, int], ...]
    tensor_snapshots: tuple[DispatchTensorSnapshot, ...] = dataclass_field(default_factory=tuple)
    push_constant_values: tuple[tuple[str, int | float], ...] = dataclass_field(default_factory=tuple)


def referenced_symbols(expr: ExprDim) -> tuple[str, ...]:
    if isinstance(expr, str):
        return (expr,)
    if isinstance(expr, int):
        return ()
    if isinstance(expr, CeilDivExpr | MulExpr | AddExpr):
        return referenced_symbols(expr.lhs) + referenced_symbols(expr.rhs)
    raise TypeError(f"Unsupported expression dim {expr!r}")


def eval_expr(expr: ExprDim, symbols: Mapping[str, int]) -> int:
    if isinstance(expr, int):
        return int(expr)
    if isinstance(expr, str):
        try:
            return int(symbols[expr])
        except KeyError as exc:
            raise ValueError(f"Unresolved shape symbol {expr!r}") from exc
    if isinstance(expr, CeilDivExpr):
        lhs = eval_expr(expr.lhs, symbols)
        rhs = eval_expr(expr.rhs, symbols)
        if rhs <= 0:
            raise ValueError(f"ceil_div rhs must be positive, got {rhs}")
        return (lhs + rhs - 1) // rhs
    if isinstance(expr, MulExpr):
        return eval_expr(expr.lhs, symbols) * eval_expr(expr.rhs, symbols)
    if isinstance(expr, AddExpr):
        return eval_expr(expr.lhs, symbols) + eval_expr(expr.rhs, symbols)
    raise TypeError(f"Unsupported expression dim {expr!r}")


def split_dynamic_push_constants(
    spec: PushConstantSpec,
    *,
    params_binding_index: int,
) -> tuple[PushConstantSpec, ParamsBufferSpec | None]:
    """Split a PushConstantSpec into static push constants and a dynamic params buffer.

    Fields with ``dynamic=True`` are moved into a ``ParamsBufferSpec`` at
    contiguous offsets.  The remaining static fields are re-packed into a new
    ``PushConstantSpec`` (offsets compacted).
    """
    static_fields: list[PushConstantFieldSpec] = []
    dynamic_fields: list[PushConstantFieldSpec] = []
    for f in spec.fields:
        if f.dynamic:
            dynamic_fields.append(f)
        else:
            static_fields.append(f)

    if not dynamic_fields:
        return spec, None

    static_offset = 0
    repacked_static: list[PushConstantFieldSpec] = []
    for f in static_fields:
        repacked_static.append(PushConstantFieldSpec(
            name=f.name,
            dtype=f.dtype,
            offset=static_offset,
            value=f.value,
            dynamic=False,
        ))
        static_offset += f.size

    params_offset = 0
    params_fields: list[ParamsBufferFieldSpec] = []
    for f in dynamic_fields:
        params_fields.append(ParamsBufferFieldSpec(
            name=f.name,
            dtype=f.dtype,
            offset=params_offset,
            value=f.value,
        ))
        params_offset += f.size

    new_push = PushConstantSpec(size=static_offset, fields=tuple(repacked_static))
    params = ParamsBufferSpec(
        size=params_offset,
        fields=tuple(params_fields),
        binding_index=params_binding_index,
    )
    return new_push, params


def _validate_push_constant_definition(shader_name: str, spec: PushConstantSpec) -> None:
    if spec.size < 0:
        raise ValueError(f"{shader_name} push constant size must be non-negative")
    names: set[str] = set()
    occupied: set[int] = set()
    for field_spec in spec.fields:
        if not field_spec.name:
            raise ValueError(f"{shader_name} has an empty push constant field name")
        if field_spec.name in names:
            raise ValueError(f"{shader_name} duplicate push constant field {field_spec.name}")
        names.add(field_spec.name)
        if field_spec.offset < 0 or field_spec.offset + field_spec.size > spec.size:
            raise ValueError(f"{shader_name}.{field_spec.name} push constant range is out of bounds")
        for byte in range(field_spec.offset, field_spec.offset + field_spec.size):
            if byte in occupied:
                raise ValueError(f"{shader_name}.{field_spec.name} overlaps another push constant")
            occupied.add(byte)
