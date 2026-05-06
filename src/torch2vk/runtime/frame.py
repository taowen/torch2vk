"""Runtime frame context state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from torch2vk.runtime.logical import LogicalTensor


@dataclass(slots=True)
class FrameContext:
    frame: str
    start_dispatch_index: int
    pytorch_model: object | None = None
    pytorch_args: tuple[object, ...] = ()
    pytorch_kwargs: Mapping[str, object] = field(default_factory=dict)
    pytorch_input_prefixes: tuple[str, ...] = ()
    pytorch_cache_policy: Literal["none", "hf_dynamic"] = "none"
    pytorch_cache_namespace: str | None = None
    pytorch_reset_cache: bool = False
    used_tensors: list[LogicalTensor] = field(default_factory=list)
    written_tensors: list[LogicalTensor] = field(default_factory=list)
    pytorch_captured_artifacts: dict[str, object] = field(default_factory=dict)
    pytorch_forward_ran: bool = False
    pytorch_cache_input_snapshots: dict[str, object] = field(default_factory=dict)
