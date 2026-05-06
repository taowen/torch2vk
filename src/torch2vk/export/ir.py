"""Export IR dataclasses shared across model exporters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorFieldPattern:
    field: str
    source_parameter: str | None = None
    note: str = ""
    dtype: str | None = None
    role: str | None = None
