"""Record/replay metadata for stable shader sequences."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .logical import LogicalTensor
from .shader import DispatchRecord


@dataclass(frozen=True, slots=True)
class ReplayRegime:
    model: str
    phase: str
    values: Mapping[str, int | str]


@dataclass(frozen=True, slots=True)
class StorageFingerprint:
    tensor_name: str
    allocation_id: str
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]
    layout: str


@dataclass(frozen=True, slots=True)
class RecordedSequence:
    regime: ReplayRegime
    dispatches: tuple[DispatchRecord, ...]
    storage: tuple[StorageFingerprint, ...] = ()

    def validate_regime(self, regime: ReplayRegime) -> None:
        if self.regime != regime:
            raise ValueError(
                f"Replay regime mismatch: expected {self.regime}, got {regime}"
            )

    def validate_storage(self, tensors: Sequence[LogicalTensor]) -> None:
        expected = Counter(self.storage)
        actual = Counter(storage_fingerprints(tensors))
        missing = expected - actual
        if missing:
            first = next(iter(missing))
            raise ValueError(f"Replay storage missing fingerprint: {first}")
        extra = actual - expected
        if extra:
            first = next(iter(extra))
            raise ValueError(f"Replay storage got extra fingerprint: {first}")


def storage_fingerprints(tensors: Sequence[LogicalTensor]) -> tuple[StorageFingerprint, ...]:
    fingerprints: list[StorageFingerprint] = []
    for tensor in tensors:
        if tensor.storage is None:
            raise ValueError(f"{tensor.name} has no storage for replay fingerprint")
        shape = _concrete_shape(tensor)
        fingerprints.append(
            StorageFingerprint(
                tensor_name=tensor.name,
                allocation_id=tensor.storage.allocation_id,
                offset=tensor.storage.offset,
                nbytes=tensor.storage.nbytes,
                dtype=tensor.dtype,
                shape=shape,
                layout=tensor.layout.name,
            )
        )
    return tuple(
        sorted(
            fingerprints,
            key=lambda item: (item.tensor_name, item.shape, item.offset, item.nbytes),
        )
    )


def _concrete_shape(tensor: LogicalTensor) -> tuple[int, ...]:
    shape: list[int] = []
    for dim in tensor.shape:
        if not isinstance(dim, int):
            raise TypeError(f"{tensor.name} has symbolic shape {tensor.shape}")
        shape.append(dim)
    return tuple(shape)
