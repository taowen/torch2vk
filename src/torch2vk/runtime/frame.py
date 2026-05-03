"""Runtime frame scope state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from torch2vk.runtime.logical import LogicalTensor


@dataclass(frozen=True, slots=True)
class FrameScope:
    frame: str
    values: Mapping[str, str | int] = field(default_factory=dict)

    def artifact_prefix(self) -> str:
        prefix = "/".join(f"{key}={self.values[key]}" for key in sorted(self.values))
        if prefix:
            return f"{prefix}/{self.frame}"
        return self.frame


@dataclass(slots=True)
class FrameContext:
    scope: FrameScope
    start_dispatch_index: int
    dependencies: tuple[LogicalTensor, ...] = ()
    pytorch_model: object | None = None
    pytorch_args: tuple[object, ...] = ()
    pytorch_kwargs: Mapping[str, object] = field(default_factory=dict)
    reference_model: object | None = None
    used_tensors: list[LogicalTensor] = field(default_factory=list)
    written_tensors: list[LogicalTensor] = field(default_factory=list)
