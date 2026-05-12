"""Generated dispatch function for run_lm_head."""

from __future__ import annotations

from models.optimized_qwen3_asr.tensors.model import model_tensors
from models.optimized_qwen3_asr.shaders.linear_nobias_q4_k_f32 import LINEAR_NOBIAS_Q4_K_F32
from models.optimized_qwen3_asr.shaders.linear_nobias_q6_k_f32 import LINEAR_NOBIAS_Q6_K_F32
from models.optimized_qwen3_asr.tensors.lm_head import LmHeadTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.shader import ShaderVariant
from torch2vk.runtime.session import RuntimeSession
from torch2vk.vulkan.types import Q4KWordsLayout, Q6KHalfwordsLayout


def _linear_q4_or_q6(
    rt: RuntimeSession,
    *,
    q4: ShaderVariant,
    q6: ShaderVariant,
    x: LogicalTensor,
    weight: LogicalTensor,
    output: LogicalTensor,
) -> None:
    if isinstance(weight.layout, Q6KHalfwordsLayout):
        q6(rt, x=x, weight=weight, output=output)
        return
    if not isinstance(weight.layout, Q4KWordsLayout):
        raise ValueError(f"{weight.name} expected Q4_K or Q6_K layout, got {weight.layout}")
    q4(rt, x=x, weight=weight, output=output)


def _run_lm_head_with_tensors(rt: RuntimeSession, tensors: LmHeadTensors) -> None:
    _linear_q4_or_q6(
        rt,
        q4=LINEAR_NOBIAS_Q4_K_F32,
        q6=LINEAR_NOBIAS_Q6_K_F32,
        x=tensors.input,
        weight=tensors.p_weight,
        output=tensors.linear,
    )


def run_lm_head(rt: RuntimeSession) -> None:
    _run_lm_head_with_tensors(rt, model_tensors().lm_head)
