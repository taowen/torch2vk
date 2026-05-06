"""Structural types used by torch2vk export."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol


class ExportOpLike(Protocol):
    @property
    def target(self) -> str: ...

    @property
    def inputs(self) -> tuple[str, ...]: ...

    @property
    def outputs(self) -> tuple[str, ...]: ...

    @property
    def name(self) -> str: ...

    @property
    def op(self) -> str: ...

    @property
    def args(self) -> tuple[object, ...]: ...

    @property
    def kwargs(self) -> tuple[tuple[str, object], ...]: ...


class FxTensorMetaLike(Protocol):
    dtype: object
    shape: Sequence[object]


class FxNodeLike(Protocol):
    op: str
    target: object
    name: str
    args: object
    kwargs: object
    meta: Mapping[str, object]


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
