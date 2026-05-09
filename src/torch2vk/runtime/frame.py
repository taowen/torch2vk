"""Runtime frame context state."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch2vk.runtime.logical import LogicalTensor


@dataclass(slots=True)
class FrameContext:
    frame: str
    start_dispatch_index: int
    end_dispatch_index: int
    registered_inputs: list[LogicalTensor] = field(default_factory=list)
    used_tensors: list[LogicalTensor] = field(default_factory=list)
    written_tensors: list[LogicalTensor] = field(default_factory=list)
