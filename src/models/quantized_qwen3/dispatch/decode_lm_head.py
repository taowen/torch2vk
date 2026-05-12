"""Generated dispatch function for run_decode_lm_head."""

from __future__ import annotations

from models.quantized_qwen3.tensors.model import model_tensors
from models.quantized_qwen3.shaders.linear_nobias_q4_k_matvec_f32 import LINEAR_NOBIAS_Q4_K_MATVEC_F32
from models.quantized_qwen3.shaders.linear_nobias_q6_k_matvec_f32 import LINEAR_NOBIAS_Q6_K_MATVEC_F32
from models.quantized_qwen3.tensors.decode_lm_head import DecodeLmHeadTensors
from torch2vk.runtime.logical import LogicalTensor
from torch2vk.runtime.shader import ShaderVariant
from torch2vk.vulkan.types import Q4KWordsLayout, Q6KHalfwordsLayout
from torch2vk.runtime.session import RuntimeSession


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


def _run_decode_lm_head_with_tensors(rt: RuntimeSession, tensors: DecodeLmHeadTensors) -> None:
    _linear_q4_or_q6(rt, q4=LINEAR_NOBIAS_Q4_K_MATVEC_F32, q6=LINEAR_NOBIAS_Q6_K_MATVEC_F32, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def run_decode_lm_head(rt: RuntimeSession) -> None:
    _run_decode_lm_head_with_tensors(rt, model_tensors().decode_lm_head)
