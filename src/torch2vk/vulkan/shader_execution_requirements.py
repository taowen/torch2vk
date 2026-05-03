"""Execution-topology requirements declared by one shader variant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ShaderComponentTypeName = Literal[
    "float16",
    "float32",
    "sint8",
    "uint8",
    "sint16",
    "uint16",
    "sint32",
    "uint32",
]
ShaderScopeName = Literal["subgroup"]


@dataclass(frozen=True, slots=True)
class SubgroupRequirements:
    required_size: int
    require_full_subgroups: bool = False

    def __post_init__(self) -> None:
        if self.required_size <= 0:
            raise ValueError(f"required_size must be positive, got {self.required_size}")


@dataclass(frozen=True, slots=True)
class CooperativeMatrixRequirements:
    scope: ShaderScopeName
    m_size: int
    n_size: int
    k_size: int
    a_type: ShaderComponentTypeName
    b_type: ShaderComponentTypeName
    c_type: ShaderComponentTypeName
    result_type: ShaderComponentTypeName
    saturating_accumulation: bool = False

    def __post_init__(self) -> None:
        if self.m_size <= 0:
            raise ValueError(f"m_size must be positive, got {self.m_size}")
        if self.n_size <= 0:
            raise ValueError(f"n_size must be positive, got {self.n_size}")
        if self.k_size <= 0:
            raise ValueError(f"k_size must be positive, got {self.k_size}")


@dataclass(frozen=True, slots=True)
class ShaderExecutionRequirements:
    subgroup: SubgroupRequirements | None = None
    cooperative_matrix: CooperativeMatrixRequirements | None = None
    require_integer_dot_product: bool = False
    require_shader_int64: bool = False
    require_buffer_device_address: bool = False
    require_storage_buffer_16bit_access: bool = False
