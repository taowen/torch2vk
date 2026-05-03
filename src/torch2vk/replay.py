"""Record/replay metadata for stable shader sequences."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
