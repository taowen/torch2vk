"""Generated dispatch function for run_text_norm."""

from __future__ import annotations

from models.exported_qwen3_asr.tensors.model import model_tensors
from models.exported_qwen3_asr.shaders.add_scalar import ADD_SCALAR
from models.exported_qwen3_asr.shaders.mean_dim_f32 import MEAN_DIM_F32
from models.exported_qwen3_asr.shaders.mul_broadcast import MUL_BROADCAST
from models.exported_qwen3_asr.shaders.mul_left_broadcast_bf16x_f32 import MUL_LEFT_BROADCAST_BF16X_F32
from models.exported_qwen3_asr.shaders.pow_scalar_f32 import POW_SCALAR_F32
from models.exported_qwen3_asr.shaders.rsqrt_f32 import RSQRT_F32
from models.exported_qwen3_asr.tensors.text_norm import TextNormTensors
from torch2vk.runtime.session import RuntimeSession


def _run_text_norm_with_tensors(rt: RuntimeSession, tensors: TextNormTensors) -> None:
    POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    MUL_BROADCAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    MUL_LEFT_BROADCAST_BF16X_F32(rt, x=tensors.p_weight, y=tensors.to_1, output=tensors.mul_1)


def run_text_norm(rt: RuntimeSession) -> None:
    tensors = model_tensors().text_norm
    _run_text_norm_with_tensors(rt, tensors)
