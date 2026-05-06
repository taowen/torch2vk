"""Structural types used by torch2vk export."""

from __future__ import annotations

from typing import Protocol


class ExportOpLike(Protocol):
    target: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    name: str
    op: str
    args: tuple[object, ...]
    kwargs: tuple[tuple[str, object], ...]
