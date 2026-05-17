"""Generated dispatch function for run_flux_final_layer."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.add_broadcast_inner import ADD_BROADCAST_INNER
from models.quantized_klein9b.shaders.flux_final_layer_add_scalar import FLUX_FINAL_LAYER_ADD_SCALAR
from models.quantized_klein9b.shaders.flux_final_layer_mul_broadcast import FLUX_FINAL_LAYER_MUL_BROADCAST
from models.quantized_klein9b.shaders.flux_final_layer_silu_f32 import FLUX_FINAL_LAYER_SILU_F32
from models.quantized_klein9b.shaders.flux_final_layer_slice_f32 import FLUX_FINAL_LAYER_SLICE_F32
from models.quantized_klein9b.shaders.flux_final_layer_tuple_getitem_slice_f32 import FLUX_FINAL_LAYER_TUPLE_GETITEM_SLICE_F32
from models.quantized_klein9b.shaders.layer_norm_nonew_noneb_f32 import LAYER_NORM_NONEW_NONEB_F32
from models.quantized_klein9b.shaders.linear_nobias_f16w_f32 import LINEAR_NOBIAS_F16W_F32
from models.quantized_klein9b.shaders.linear_nobias_f16w_f32_9 import LINEAR_NOBIAS_F16W_F32_9
from models.quantized_klein9b.shaders.tuple_getitem_slice_f32_4 import TUPLE_GETITEM_SLICE_F32_4
from models.quantized_klein9b.tensors.flux_final_layer import FluxFinalLayerTensors
from torch2vk.runtime.session import RuntimeSession


def _run_flux_final_layer_with_tensors(rt: RuntimeSession, tensors: FluxFinalLayerTensors) -> None:
    FLUX_FINAL_LAYER_SLICE_F32(rt, x=tensors.hidden_states, output=tensors.slice_1)
    FLUX_FINAL_LAYER_SILU_F32(rt, x=tensors.vec, output=tensors.silu)
    LINEAR_NOBIAS_F16W_F32(rt, x=tensors.silu, weight=tensors.p_final_layer_adaln_modulation_1_weight, output=tensors.linear)
    FLUX_FINAL_LAYER_TUPLE_GETITEM_SLICE_F32(rt, x=tensors.linear, output=tensors.getitem)
    TUPLE_GETITEM_SLICE_F32_4(rt, x=tensors.linear, output=tensors.getitem_1)
    FLUX_FINAL_LAYER_ADD_SCALAR(rt, x=tensors.unsqueeze_1, output=tensors.add)
    LAYER_NORM_NONEW_NONEB_F32(rt, x=tensors.slice_1, output=tensors.layer_norm)
    FLUX_FINAL_LAYER_MUL_BROADCAST(rt, x=tensors.add, y=tensors.layer_norm, output=tensors.mul)
    ADD_BROADCAST_INNER(rt, x=tensors.mul, y=tensors.unsqueeze, output=tensors.add_1)
    LINEAR_NOBIAS_F16W_F32_9(rt, x=tensors.add_1, weight=tensors.p_final_layer_linear_weight, output=tensors.linear_1)


def run_flux_final_layer(rt: RuntimeSession) -> None:
    tensors = model_tensors().flux_final_layer
    _run_flux_final_layer_with_tensors(rt, tensors)
