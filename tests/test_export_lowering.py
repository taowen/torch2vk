"""Coverage for export lowering registry constraints."""

from __future__ import annotations

import pytest

from torch2vk.export.ir import TorchOpPattern
from torch2vk.export.lowering import OpLoweringRegistry, OpShaderBinding


def test_lowering_registry_rejects_non_aten_binding_target() -> None:
    registry = OpLoweringRegistry()
    with pytest.raises(ValueError, match="must start with 'aten\\.'"):
        registry.register(OpShaderBinding(target="linear", shader="SOME_SHADER"))


def test_lowering_registry_rejects_non_aten_op_target_on_resolve() -> None:
    registry = OpLoweringRegistry(
        (OpShaderBinding(target="aten.add.Tensor", shader="ADD_SHADER"),)
    )
    op = TorchOpPattern(target="linear", inputs=(), outputs=())
    with pytest.raises(ValueError, match="must start with 'aten\\.'"):
        registry.resolve(op=op)


def test_lowering_registry_resolves_aten_target() -> None:
    registry = OpLoweringRegistry(
        (OpShaderBinding(target="aten.add.Tensor", shader="ADD_SHADER"),)
    )
    op = TorchOpPattern(target="aten.add.Tensor", inputs=("x", "y"), outputs=("z",))
    binding = registry.resolve(op=op)
    assert binding is not None
    assert binding.shader == "ADD_SHADER"

