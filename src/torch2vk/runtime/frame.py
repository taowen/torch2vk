"""Runtime frame scope state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


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
    pytorch_model: object | None = None
    reference_model: object | None = None
