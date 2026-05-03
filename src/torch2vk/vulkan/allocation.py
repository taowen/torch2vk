"""Vulkan buffer allocation ownership and slice primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


class BufferOwner(Protocol):
    handle: object
    memory: object
    size: int

    def map_persistent(self) -> None: ...

    def write_bytes_at(self, offset: int, data: bytes | bytearray | memoryview) -> None: ...

    def read_bytes_at(self, offset: int, size: int) -> bytes: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class GpuTimelinePoint:
    queue_id: str
    submit_id: int
    fence: object


@dataclass(slots=True)
class BufferAllocation:
    buffer: BufferOwner
    pool: str
    offset: int = 0
    size_bytes: int | None = None
    vk_allocation: bool = False
    releaser: Callable[["BufferAllocation"], None] | None = None
    last_use: GpuTimelinePoint | None = None
    _released: bool = False

    @property
    def size(self) -> int:
        return self.buffer.size if self.size_bytes is None else int(self.size_bytes)

    @property
    def end_offset(self) -> int:
        return self.offset + self.size

    @property
    def released(self) -> bool:
        return self._released

    def close(self) -> None:
        if self._released:
            return
        self._released = True
        if self.releaser is not None:
            self.releaser(self)
            return
        self.buffer.close()


@dataclass(frozen=True, slots=True)
class BufferSlice:
    allocation: BufferAllocation
    offset: int
    nbytes: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError(f"BufferSlice offset must be non-negative, got {self.offset}")
        if self.nbytes <= 0:
            raise ValueError(f"BufferSlice nbytes must be positive, got {self.nbytes}")
        if self.offset < self.allocation.offset or self.offset + self.nbytes > self.allocation.end_offset:
            raise ValueError(
                f"BufferSlice range [{self.offset}, {self.offset + self.nbytes}) exceeds allocation range "
                f"[{self.allocation.offset}, {self.allocation.end_offset})"
            )
