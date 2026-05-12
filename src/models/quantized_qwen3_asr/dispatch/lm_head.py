"""Generated dispatch function for run_lm_head."""

from __future__ import annotations

from models.quantized_qwen3_asr.tensors.model import model_tensors
from models.quantized_qwen3_asr.shaders.linear_nobias_q4_k_matvec_f32_act_f32 import LINEAR_NOBIAS_Q4_K_MATVEC_F32_ACT_F32
from models.quantized_qwen3_asr.shaders.linear_nobias_q6_k_matvec_f32_act_f32 import LINEAR_NOBIAS_Q6_K_MATVEC_F32_ACT_F32
from models.quantized_qwen3_asr.shaders.linear_nobias_q8_0_matvec_f32_act_f32 import LINEAR_NOBIAS_Q8_0_MATVEC_F32_ACT_F32
from models.quantized_qwen3_asr.tensors.lm_head import LmHeadTensors
from torch2vk.runtime.quantized_dispatch import run_quantized_linear
from torch2vk.runtime.session import RuntimeSession


def _run_lm_head_with_tensors(rt: RuntimeSession, tensors: LmHeadTensors) -> None:
    run_quantized_linear(rt, q4=LINEAR_NOBIAS_Q4_K_MATVEC_F32_ACT_F32, q6=LINEAR_NOBIAS_Q6_K_MATVEC_F32_ACT_F32, q8=LINEAR_NOBIAS_Q8_0_MATVEC_F32_ACT_F32, x=tensors.input, weight=tensors.p_weight, output=tensors.linear)


def run_lm_head(rt: RuntimeSession) -> None:
    _run_lm_head_with_tensors(rt, model_tensors().lm_head)
