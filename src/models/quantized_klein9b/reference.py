"""Generated PyTorch reference comparison functions."""

from __future__ import annotations

import numpy as np
import torch

from models.quantized_klein9b.dispatch.flux_prologue import run_flux_prologue as _dispatch_flux_prologue
from models.quantized_klein9b.dispatch.flux_double_block import run_flux_double_block as _dispatch_flux_double_block
from models.quantized_klein9b.dispatch.flux_single_block import run_flux_single_block as _dispatch_flux_single_block
from models.quantized_klein9b.dispatch.flux_final_layer import run_flux_final_layer as _dispatch_flux_final_layer
from models.quantized_klein9b.tensors.model import model_tensors
from torch2vk.runtime.compare import as_numpy_array
from torch2vk.runtime.logical import ComparePolicy
from torch2vk.runtime.session import RuntimeSession
from torch2vk.runtime.streaming_compare import compare_vulkan_stage


ReferenceInput = np.ndarray | torch.Tensor | int
ReferenceExpected = dict[str, object]

_COMPARE_POLICIES = {
    "tensor": ComparePolicy(kind="tensor", rtol=1e-2, atol=1.5),
    "q8_tensor": ComparePolicy(kind="tensor", rtol=2e-2, atol=8.0),
    "q4_tensor": ComparePolicy(kind="tensor", rtol=5e-2, atol=128.0),
    "token": ComparePolicy(kind="token"),
}


def _reference_int(value: ReferenceInput) -> int:
    if isinstance(value, int):
        return value
    return int(as_numpy_array(value).reshape(-1)[0])


def _policy(policy: str | dict[str, str]) -> ComparePolicy | dict[str, ComparePolicy]:
    if isinstance(policy, dict):
        return {key: _COMPARE_POLICIES[value] for key, value in policy.items()}
    return _COMPARE_POLICIES[policy]

def compare_flux_prologue(
    rt: RuntimeSession,
    *,
    step: int,
    expected: ReferenceExpected,
    x: ReferenceInput,
    x_ids: ReferenceInput,
    timesteps: ReferenceInput,
    ctx: ReferenceInput,
    ctx_ids: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name=f'klein9b.flux.compare.{step:04d}.prologue',
        run=lambda: _dispatch_flux_prologue(rt),
        tensors=model_tensors().flux_prologue,
        input_bindings={'x': 'x', 'x_ids': 'x_ids', 'timesteps': 'timesteps', 'ctx': 'ctx', 'ctx_ids': 'ctx_ids'},
        output_bindings={'img': 'linear_5', 'txt': 'linear_6', 'pe_x': 'unsqueeze_5', 'pe_ctx': 'unsqueeze_6', 'vec': 'linear_1', 'img_mod1_shift': 'getitem', 'img_mod1_scale': 'getitem_1', 'img_mod1_gate': 'getitem_2', 'img_mod2_shift': 'getitem_3', 'img_mod2_scale': 'getitem_4', 'img_mod2_gate': 'getitem_5', 'txt_mod1_shift': 'getitem_6', 'txt_mod1_scale': 'getitem_7', 'txt_mod1_gate': 'getitem_8', 'txt_mod2_shift': 'getitem_9', 'txt_mod2_scale': 'getitem_10', 'txt_mod2_gate': 'getitem_11', 'single_mod_shift': 'getitem_12', 'single_mod_scale': 'getitem_13', 'single_mod_gate': 'getitem_14'},
        policy=_policy('q8_tensor'),
        inputs={
            "x": x,
            "x_ids": x_ids,
            "timesteps": timesteps,
            "ctx": ctx,
            "ctx_ids": ctx_ids,
        },
        expected=expected,
    )

def compare_flux_double_block(
    rt: RuntimeSession,
    *,
    step: int,
    layer_idx: int,
    expected: ReferenceExpected,
    img: ReferenceInput,
    txt: ReferenceInput,
    pe: ReferenceInput,
    pe_ctx: ReferenceInput,
    img_mod1_shift: ReferenceInput,
    img_mod1_scale: ReferenceInput,
    img_mod1_gate: ReferenceInput,
    img_mod2_shift: ReferenceInput,
    img_mod2_scale: ReferenceInput,
    img_mod2_gate: ReferenceInput,
    txt_mod1_shift: ReferenceInput,
    txt_mod1_scale: ReferenceInput,
    txt_mod1_gate: ReferenceInput,
    txt_mod2_shift: ReferenceInput,
    txt_mod2_scale: ReferenceInput,
    txt_mod2_gate: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name=f'klein9b.flux.compare.{step:04d}.double_block.{layer_idx}',
        run=lambda: _dispatch_flux_double_block(rt, layer_idx),
        tensors=model_tensors().flux_double_blocks[layer_idx],
        input_bindings={'img': 'img', 'txt': 'txt', 'pe': 'pe', 'pe_ctx': 'pe_ctx', 'img_mod1_shift': 'img_mod1_shift', 'img_mod1_scale': 'img_mod1_scale', 'img_mod1_gate': 'img_mod1_gate', 'img_mod2_shift': 'img_mod2_shift', 'img_mod2_scale': 'img_mod2_scale', 'img_mod2_gate': 'img_mod2_gate', 'txt_mod1_shift': 'txt_mod1_shift', 'txt_mod1_scale': 'txt_mod1_scale', 'txt_mod1_gate': 'txt_mod1_gate', 'txt_mod2_shift': 'txt_mod2_shift', 'txt_mod2_scale': 'txt_mod2_scale', 'txt_mod2_gate': 'txt_mod2_gate'},
        output_bindings={'img': 'add_15', 'txt': 'add_19'},
        policy=_policy('q8_tensor'),
        inputs={
            "img": img,
            "txt": txt,
            "pe": pe,
            "pe_ctx": pe_ctx,
            "img_mod1_shift": img_mod1_shift,
            "img_mod1_scale": img_mod1_scale,
            "img_mod1_gate": img_mod1_gate,
            "img_mod2_shift": img_mod2_shift,
            "img_mod2_scale": img_mod2_scale,
            "img_mod2_gate": img_mod2_gate,
            "txt_mod1_shift": txt_mod1_shift,
            "txt_mod1_scale": txt_mod1_scale,
            "txt_mod1_gate": txt_mod1_gate,
            "txt_mod2_shift": txt_mod2_shift,
            "txt_mod2_scale": txt_mod2_scale,
            "txt_mod2_gate": txt_mod2_gate,
        },
        expected=expected,
    )

def compare_flux_single_block(
    rt: RuntimeSession,
    *,
    step: int,
    layer_idx: int,
    expected: ReferenceExpected,
    hidden_states: ReferenceInput,
    pe: ReferenceInput,
    mod_shift: ReferenceInput,
    mod_scale: ReferenceInput,
    mod_gate: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name=f'klein9b.flux.compare.{step:04d}.single_block.{layer_idx}',
        run=lambda: _dispatch_flux_single_block(rt, layer_idx),
        tensors=model_tensors().flux_single_blocks[layer_idx],
        input_bindings={'hidden_states': 'hidden_states', 'pe': 'pe', 'mod_shift': 'mod_shift', 'mod_scale': 'mod_scale', 'mod_gate': 'mod_gate'},
        output_bindings={'hidden_states': 'add_6'},
        policy=_policy('q8_tensor'),
        inputs={
            "hidden_states": hidden_states,
            "pe": pe,
            "mod_shift": mod_shift,
            "mod_scale": mod_scale,
            "mod_gate": mod_gate,
        },
        expected=expected,
    )

def compare_flux_final_layer(
    rt: RuntimeSession,
    *,
    step: int,
    expected: ReferenceExpected,
    hidden_states: ReferenceInput,
    vec: ReferenceInput,
) -> ReferenceExpected:
    return compare_vulkan_stage(
        rt,
        name=f'klein9b.flux.compare.{step:04d}.final_layer',
        run=lambda: _dispatch_flux_final_layer(rt),
        tensors=model_tensors().flux_final_layer,
        input_bindings={'hidden_states': 'hidden_states', 'vec': 'vec'},
        output_bindings={'pred': 'linear_1'},
        policy=_policy('q8_tensor'),
        inputs={
            "hidden_states": hidden_states,
            "vec": vec,
        },
        expected=expected,
    )
