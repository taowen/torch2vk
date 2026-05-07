"""Structural types used by exportv2."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol


class FxNodeLike(Protocol):
    op: str
    target: object
    name: str
    args: object
    kwargs: object
    meta: Mapping[str, object]


class ExportOpLike(Protocol):
    target: str
    inputs: Sequence[str]
    outputs: Sequence[str]


class FxGraphLike(Protocol):
    nodes: Iterable[FxNodeLike]


class ExportGraphArgumentLike(Protocol):
    name: str


class ExportGraphInputSpecLike(Protocol):
    arg: ExportGraphArgumentLike
    target: object | None
    kind: object


class ExportGraphOutputSpecLike(Protocol):
    arg: ExportGraphArgumentLike | None
    kind: object


class ExportGraphSignatureLike(Protocol):
    input_specs: Sequence[ExportGraphInputSpecLike]
    output_specs: Sequence[ExportGraphOutputSpecLike]


class ExportedProgramLike(Protocol):
    graph: FxGraphLike
    graph_signature: ExportGraphSignatureLike | None
