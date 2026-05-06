"""Export IR dataclasses shared across model exporters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorFieldPattern:
    field: str
    source_parameter: str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class TorchOpPattern:
    target: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    note: str = ""
    name: str = ""
    op: str = "call_function"
    args: tuple[object, ...] = ()
    kwargs: tuple[tuple[str, object], ...] = ()
    shape: tuple[int, ...] | None = None
    dtype: str | None = None

