"""Framework-level op->shader lowering registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from torch2vk.export.ir import TorchOpPattern


LoweringMatchFn = Callable[[TorchOpPattern], bool]


@dataclass(frozen=True, slots=True)
class OpShaderBinding:
    model: str
    frame: str
    target: str
    shader: str
    match: LoweringMatchFn | None = None
    note: str = ""

    def matches(self, op: TorchOpPattern) -> bool:
        if op.target != self.target:
            return False
        if self.match is None:
            return True
        return self.match(op)


class OpLoweringRegistry:
    def __init__(self, bindings: Iterable[OpShaderBinding] = ()) -> None:
        self._bindings = list(bindings)

    def register(self, binding: OpShaderBinding) -> None:
        self._bindings.append(binding)

    def resolve(self, *, model: str, frame: str, op: TorchOpPattern) -> OpShaderBinding | None:
        for binding in self._bindings:
            if binding.model != model or binding.frame != frame:
                continue
            if binding.matches(op):
                return binding
        return None


def _match_second_input(name: str) -> LoweringMatchFn:
    def _match(op: TorchOpPattern) -> bool:
        return len(op.inputs) > 1 and op.inputs[1] == name

    return _match


DEFAULT_LOWERING_REGISTRY = OpLoweringRegistry(
    (
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.select.int",
            shader="OMNIVOICE_ATEN_SELECT_INT_I64",
        ),
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.embedding.default",
            shader="OMNIVOICE_ATEN_EMBEDDING_F32",
            match=_match_second_input("text_token_ids"),
        ),
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.add.Tensor",
            shader="OMNIVOICE_ATEN_SHIFTED_IDS_I64",
        ),
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.embedding.default",
            shader="OMNIVOICE_ATEN_EMBEDDING_3D_F32",
            match=_match_second_input("shifted_ids"),
        ),
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.sum.dim_IntList",
            shader="OMNIVOICE_ATEN_SUM_DIM1_F32",
        ),
        OpShaderBinding(
            model="omnivoice",
            frame="input_embeddings",
            target="aten.where.self",
            shader="OMNIVOICE_ATEN_WHERE_F32",
        ),
    )
)


def resolve_shader_symbol(*, model: str, frame: str, op: TorchOpPattern) -> str | None:
    binding = DEFAULT_LOWERING_REGISTRY.resolve(model=model, frame=frame, op=op)
    return None if binding is None else binding.shader

