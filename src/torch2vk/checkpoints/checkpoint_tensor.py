"""Typed mmap-backed checkpoint tensor views."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.vulkan.types import CONTIGUOUS_LAYOUT, Residency, TensorLayout, TensorSpec, concrete_nbytes

from .gguf import GGUFMmap
from .safetensors import SafetensorsMmap


@dataclass(frozen=True, slots=True)
class CheckpointTensor:
    storage: SafetensorsMmap | GGUFMmap
    tensor_key: str
    logical_spec: TensorSpec
    physical_spec: TensorSpec
    logical_layout: TensorLayout = CONTIGUOUS_LAYOUT
    physical_layout: TensorLayout = CONTIGUOUS_LAYOUT
    byte_offset: int = 0
    byte_size: int = 0
    row_offset: int | None = None
    row_count: int | None = None

    @property
    def spec(self) -> TensorSpec:
        return self.logical_spec

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.logical_spec.shape)

    @property
    def dtype(self) -> str:
        return self.logical_spec.dtype

    @property
    def nbytes(self) -> int:
        return self.byte_size

    @property
    def physical_key(self) -> tuple[str, str]:
        return (str(self.storage.path), self.tensor_key)

    @classmethod
    def open(
        cls,
        *,
        storage: SafetensorsMmap | GGUFMmap,
        tensor_key: str,
        dtype: str,
        shape: tuple[int, ...],
        layout: TensorLayout = CONTIGUOUS_LAYOUT,
    ) -> "CheckpointTensor":
        logical_spec = TensorSpec(
            dtype=dtype,
            shape=shape,
            residency=Residency.HOST,
        )
        physical_spec = TensorSpec(
            dtype=dtype,
            shape=shape,
            residency=Residency.HOST,
        )
        tensor = cls(
            storage=storage,
            tensor_key=tensor_key,
            logical_spec=logical_spec,
            physical_spec=physical_spec,
            logical_layout=layout,
            physical_layout=CONTIGUOUS_LAYOUT,
            byte_size=concrete_nbytes(dtype=dtype, shape=shape),
        )
        tensor.validate()
        return tensor

    @classmethod
    def open_row_slice(
        cls,
        *,
        storage: SafetensorsMmap | GGUFMmap,
        tensor_key: str,
        dtype: str,
        logical_shape: tuple[int, ...],
        physical_shape: tuple[int, ...],
        row_offset: int,
        row_count: int,
        layout: TensorLayout = CONTIGUOUS_LAYOUT,
    ) -> "CheckpointTensor":
        logical_spec = TensorSpec(
            dtype=dtype,
            shape=logical_shape,
            residency=Residency.HOST,
        )
        physical_spec = TensorSpec(
            dtype=dtype,
            shape=physical_shape,
            residency=Residency.HOST,
        )
        tensor = cls(
            storage=storage,
            tensor_key=tensor_key,
            logical_spec=logical_spec,
            physical_spec=physical_spec,
            logical_layout=layout,
            physical_layout=CONTIGUOUS_LAYOUT,
            byte_offset=row_offset * _row_nbytes(dtype=dtype, physical_shape=physical_shape),
            byte_size=concrete_nbytes(dtype=dtype, shape=logical_shape),
            row_offset=row_offset,
            row_count=row_count,
        )
        tensor.validate()
        return tensor

    def validate(self) -> None:
        self._validate_entry(spec=self.logical_spec, physical_spec=self.physical_spec)
        if self.row_offset is not None and self.row_count is None:
            raise ValueError(f"{self.tensor_key} row slicing requires row_count")

    def load(self) -> "CheckpointTensor":
        self.validate()
        return self

    def buffer_view(self) -> memoryview:
        self.validate()
        if self.row_offset is None:
            return self.storage.buffer_slice(self.tensor_key)
        if self.row_count is None:
            raise ValueError(f"{self.tensor_key} row slicing requires row_count")
        return self.storage.buffer_rows(
            self.tensor_key,
            row_offset=self.row_offset,
            row_count=self.row_count,
        )

    def physical_buffer_view(self) -> memoryview:
        self._validate_entry(spec=self.physical_spec, physical_spec=self.physical_spec)
        return self.storage.buffer_slice(self.tensor_key)

    def _validate_entry(self, *, spec: TensorSpec, physical_spec: TensorSpec) -> None:
        entry = self.storage.entry(self.tensor_key)
        if entry.spec.dtype != spec.dtype:
            raise ValueError(
                f"{self.tensor_key} expects dtype {spec.dtype}, got {entry.spec.dtype}"
            )

        actual_full_shape = tuple(int(dim) for dim in entry.spec.shape)
        expected_full_shape = tuple(int(dim) for dim in physical_spec.shape)
        if actual_full_shape != expected_full_shape:
            raise ValueError(
                f"{self.tensor_key} expects physical shape {expected_full_shape}, got {actual_full_shape}"
            )


def _row_nbytes(*, dtype: str, physical_shape: tuple[int, ...]) -> int:
    if not physical_shape:
        raise ValueError("row slicing requires rank >= 1 tensor")
    trailing_shape = tuple(int(dim) for dim in physical_shape[1:])
    return concrete_nbytes(dtype=dtype, shape=trailing_shape or (1,))
