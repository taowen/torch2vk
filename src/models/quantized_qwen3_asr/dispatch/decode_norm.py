"""Generated dispatch function for run_decode_norm."""

from __future__ import annotations

from models.quantized_qwen3_asr.tensors.model import model_tensors
from models.quantized_qwen3_asr.shaders.add_scalar import ADD_SCALAR
from models.quantized_qwen3_asr.shaders.decode_norm_mul_broadcast_last import DECODE_NORM_MUL_BROADCAST_LAST
from models.quantized_qwen3_asr.shaders.mean_dim_f32 import MEAN_DIM_F32
from models.quantized_qwen3_asr.shaders.mul_left_broadcast_f32x_f32 import MUL_LEFT_BROADCAST_F32X_F32
from models.quantized_qwen3_asr.shaders.pow_scalar_f32 import POW_SCALAR_F32
from models.quantized_qwen3_asr.shaders.rsqrt_f32 import RSQRT_F32
from models.quantized_qwen3_asr.tensors.decode_norm import DecodeNormTensors
from torch2vk.runtime.session import RuntimeSession


def _run_decode_norm_with_tensors(rt: RuntimeSession, tensors: DecodeNormTensors) -> None:
    POW_SCALAR_F32(rt, x=tensors.to, output=tensors.pow_1)
    MEAN_DIM_F32(rt, x=tensors.pow_1, output=tensors.mean)
    ADD_SCALAR(rt, x=tensors.mean, output=tensors.add)
    RSQRT_F32(rt, x=tensors.add, output=tensors.rsqrt)
    DECODE_NORM_MUL_BROADCAST_LAST(rt, x=tensors.to, y=tensors.rsqrt, output=tensors.mul)
    MUL_LEFT_BROADCAST_F32X_F32(rt, x=tensors.p_weight, y=tensors.to_1, output=tensors.mul_1)


def run_decode_norm(rt: RuntimeSession) -> None:
    _run_decode_norm_with_tensors(rt, model_tensors().decode_norm)
