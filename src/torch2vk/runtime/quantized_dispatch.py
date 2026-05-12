"""Runtime dispatch helpers for quantized generated shaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.shader import ShaderVariant
from torch2vk.vulkan.types import Q4KWordsLayout, Q6KHalfwordsLayout, Q8_0HalfwordsLayout

if TYPE_CHECKING:
    from torch2vk.runtime.session import RuntimeSession


def run_quantized_linear(
    rt: RuntimeSession,
    *,
    q4: ShaderVariant,
    q6: ShaderVariant,
    q8: ShaderVariant,
    x: LogicalTensor,
    weight: LogicalTensor,
    output: LogicalTensor,
) -> None:
    if isinstance(weight.layout, Q6KHalfwordsLayout):
        q6(rt, x=x, weight=weight, output=output)
        return
    if isinstance(weight.layout, Q8_0HalfwordsLayout):
        q8(rt, x=x, weight=weight, output=output)
        return
    if not isinstance(weight.layout, Q4KWordsLayout):
        raise ValueError(
            f"{weight.name} expected Q4_K, Q6_K, or Q8_0 layout, got {weight.layout}"
        )
    q4(rt, x=x, weight=weight, output=output)
