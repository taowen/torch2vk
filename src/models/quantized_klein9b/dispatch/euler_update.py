"""Generated dispatch function for run_euler_update."""

from __future__ import annotations

from models.quantized_klein9b.tensors.model import model_tensors
from models.quantized_klein9b.shaders.euler_update_add_f32 import EULER_UPDATE_ADD_F32
from models.quantized_klein9b.shaders.euler_update_mul_broadcast import EULER_UPDATE_MUL_BROADCAST
from models.quantized_klein9b.tensors.euler_update import EulerUpdateTensors
from torch2vk.runtime.session import RuntimeSession


def _run_euler_update_with_tensors(rt: RuntimeSession, tensors: EulerUpdateTensors) -> None:
    EULER_UPDATE_MUL_BROADCAST(rt, x=tensors.view, y=tensors.pred, output=tensors.mul)
    EULER_UPDATE_ADD_F32(rt, x=tensors.x, y=tensors.mul, output=tensors.add)
    rt.release_frame_workspace(tensors.mul)


def run_euler_update(rt: RuntimeSession) -> None:
    tensors = model_tensors().euler_update
    _run_euler_update_with_tensors(rt, tensors)
