"""Runtime frame context state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from torch2vk.runtime.logical import LogicalTensor


@dataclass(slots=True)
class FrameContext:
    frame: str
    start_dispatch_index: int
    pytorch_model: object | None = None
    pytorch_args: tuple[object, ...] = ()
    pytorch_kwargs: Mapping[str, object] = field(default_factory=dict)
    reference_model: object | None = None
    used_tensors: list[LogicalTensor] = field(default_factory=list)
    written_tensors: list[LogicalTensor] = field(default_factory=list)
