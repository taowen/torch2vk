"""Generated tensor declarations."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from torch2vk.runtime.logical import (
    LogicalTensor,
    MemoryClass,
    TensorLifetime,
    TensorRole,
    bind_logical_tensor_alias,
    bind_logical_tensor_names,
)
from torch2vk.vulkan.types import (
    CONTIGUOUS_LAYOUT,
    TensorLayout,
    TensorSpec,
    q4_k_words_layout,
    q6_k_halfwords_layout,
    q8_0_halfwords_layout,
)


@dataclass(frozen=True, slots=True)
class FluxPrologueTensors:
    p_img_in_weight: LogicalTensor
    p_time_in_in_layer_weight: LogicalTensor
    p_time_in_out_layer_weight: LogicalTensor
    p_txt_in_weight: LogicalTensor
    p_double_stream_modulation_img_lin_weight: LogicalTensor
    p_double_stream_modulation_txt_lin_weight: LogicalTensor
    p_single_stream_modulation_lin_weight: LogicalTensor
    x: LogicalTensor
    x_ids: LogicalTensor
    timesteps: LogicalTensor
    ctx: LogicalTensor
    ctx_ids: LogicalTensor
    mul: LogicalTensor
    arange: LogicalTensor
    mul_1: LogicalTensor
    div: LogicalTensor
    exp: LogicalTensor
    unsqueeze: LogicalTensor
    to: LogicalTensor
    unsqueeze_1: LogicalTensor
    mul_2: LogicalTensor
    cos: LogicalTensor
    sin: LogicalTensor
    cat: LogicalTensor
    to_1: LogicalTensor
    linear: LogicalTensor
    silu: LogicalTensor
    linear_1: LogicalTensor
    silu_1: LogicalTensor
    linear_2: LogicalTensor
    unsqueeze_2: LogicalTensor
    getitem: LogicalTensor
    getitem_1: LogicalTensor
    getitem_2: LogicalTensor
    getitem_3: LogicalTensor
    getitem_4: LogicalTensor
    getitem_5: LogicalTensor
    silu_2: LogicalTensor
    linear_3: LogicalTensor
    unsqueeze_3: LogicalTensor
    getitem_6: LogicalTensor
    getitem_7: LogicalTensor
    getitem_8: LogicalTensor
    getitem_9: LogicalTensor
    getitem_10: LogicalTensor
    getitem_11: LogicalTensor
    silu_3: LogicalTensor
    linear_4: LogicalTensor
    unsqueeze_4: LogicalTensor
    getitem_12: LogicalTensor
    getitem_13: LogicalTensor
    getitem_14: LogicalTensor
    linear_5: LogicalTensor
    linear_6: LogicalTensor
    select: LogicalTensor
    arange_1: LogicalTensor
    div_1: LogicalTensor
    pow_1: LogicalTensor
    reciprocal: LogicalTensor
    mul_3: LogicalTensor
    einsum: LogicalTensor
    cos_1: LogicalTensor
    sin_1: LogicalTensor
    neg: LogicalTensor
    sin_2: LogicalTensor
    cos_2: LogicalTensor
    stack: LogicalTensor
    reshape: LogicalTensor
    to_2: LogicalTensor
    select_1: LogicalTensor
    arange_2: LogicalTensor
    div_2: LogicalTensor
    pow_2: LogicalTensor
    reciprocal_1: LogicalTensor
    mul_4: LogicalTensor
    einsum_1: LogicalTensor
    cos_3: LogicalTensor
    sin_3: LogicalTensor
    neg_1: LogicalTensor
    sin_4: LogicalTensor
    cos_4: LogicalTensor
    stack_1: LogicalTensor
    reshape_1: LogicalTensor
    to_3: LogicalTensor
    select_2: LogicalTensor
    arange_3: LogicalTensor
    div_3: LogicalTensor
    pow_3: LogicalTensor
    reciprocal_2: LogicalTensor
    mul_5: LogicalTensor
    einsum_2: LogicalTensor
    cos_5: LogicalTensor
    sin_5: LogicalTensor
    neg_2: LogicalTensor
    sin_6: LogicalTensor
    cos_6: LogicalTensor
    stack_2: LogicalTensor
    reshape_2: LogicalTensor
    to_4: LogicalTensor
    select_3: LogicalTensor
    arange_4: LogicalTensor
    div_4: LogicalTensor
    pow_4: LogicalTensor
    reciprocal_3: LogicalTensor
    mul_6: LogicalTensor
    einsum_3: LogicalTensor
    cos_7: LogicalTensor
    sin_7: LogicalTensor
    neg_3: LogicalTensor
    sin_8: LogicalTensor
    cos_8: LogicalTensor
    stack_3: LogicalTensor
    reshape_3: LogicalTensor
    to_5: LogicalTensor
    cat_1: LogicalTensor
    unsqueeze_5: LogicalTensor
    select_4: LogicalTensor
    arange_5: LogicalTensor
    div_5: LogicalTensor
    pow_5: LogicalTensor
    reciprocal_4: LogicalTensor
    mul_7: LogicalTensor
    einsum_4: LogicalTensor
    cos_9: LogicalTensor
    sin_9: LogicalTensor
    neg_4: LogicalTensor
    sin_10: LogicalTensor
    cos_10: LogicalTensor
    stack_4: LogicalTensor
    reshape_4: LogicalTensor
    to_6: LogicalTensor
    select_5: LogicalTensor
    arange_6: LogicalTensor
    div_6: LogicalTensor
    pow_6: LogicalTensor
    reciprocal_5: LogicalTensor
    mul_8: LogicalTensor
    einsum_5: LogicalTensor
    cos_11: LogicalTensor
    sin_11: LogicalTensor
    neg_5: LogicalTensor
    sin_12: LogicalTensor
    cos_12: LogicalTensor
    stack_5: LogicalTensor
    reshape_5: LogicalTensor
    to_7: LogicalTensor
    select_6: LogicalTensor
    arange_7: LogicalTensor
    div_7: LogicalTensor
    pow_7: LogicalTensor
    reciprocal_6: LogicalTensor
    mul_9: LogicalTensor
    einsum_6: LogicalTensor
    cos_13: LogicalTensor
    sin_13: LogicalTensor
    neg_6: LogicalTensor
    sin_14: LogicalTensor
    cos_14: LogicalTensor
    stack_6: LogicalTensor
    reshape_6: LogicalTensor
    to_8: LogicalTensor
    select_7: LogicalTensor
    arange_8: LogicalTensor
    div_8: LogicalTensor
    pow_8: LogicalTensor
    reciprocal_7: LogicalTensor
    mul_10: LogicalTensor
    einsum_7: LogicalTensor
    cos_15: LogicalTensor
    sin_15: LogicalTensor
    neg_7: LogicalTensor
    sin_16: LogicalTensor
    cos_16: LogicalTensor
    stack_7: LogicalTensor
    reshape_7: LogicalTensor
    to_9: LogicalTensor
    cat_2: LogicalTensor
    unsqueeze_6: LogicalTensor


FLUX_PROLOGUE_OUTPUT: str = 'linear_5'


def create_flux_prologue(
    prefix: str,
    *,
    image_seq_len: int,
    text_seq_len: int,
    p_img_in_weight: LogicalTensor | None = None,
    p_time_in_in_layer_weight: LogicalTensor | None = None,
    p_time_in_out_layer_weight: LogicalTensor | None = None,
    p_txt_in_weight: LogicalTensor | None = None,
    p_double_stream_modulation_img_lin_weight: LogicalTensor | None = None,
    p_double_stream_modulation_txt_lin_weight: LogicalTensor | None = None,
    p_single_stream_modulation_lin_weight: LogicalTensor | None = None,
    x: LogicalTensor | None = None,
    x_ids: LogicalTensor | None = None,
    timesteps: LogicalTensor | None = None,
    ctx: LogicalTensor | None = None,
    ctx_ids: LogicalTensor | None = None,
    mul: LogicalTensor | None = None,
    arange: LogicalTensor | None = None,
    mul_1: LogicalTensor | None = None,
    div: LogicalTensor | None = None,
    exp: LogicalTensor | None = None,
    unsqueeze: LogicalTensor | None = None,
    to: LogicalTensor | None = None,
    unsqueeze_1: LogicalTensor | None = None,
    mul_2: LogicalTensor | None = None,
    cos: LogicalTensor | None = None,
    sin: LogicalTensor | None = None,
    cat: LogicalTensor | None = None,
    to_1: LogicalTensor | None = None,
    linear: LogicalTensor | None = None,
    silu: LogicalTensor | None = None,
    linear_1: LogicalTensor | None = None,
    silu_1: LogicalTensor | None = None,
    linear_2: LogicalTensor | None = None,
    unsqueeze_2: LogicalTensor | None = None,
    getitem: LogicalTensor | None = None,
    getitem_1: LogicalTensor | None = None,
    getitem_2: LogicalTensor | None = None,
    getitem_3: LogicalTensor | None = None,
    getitem_4: LogicalTensor | None = None,
    getitem_5: LogicalTensor | None = None,
    silu_2: LogicalTensor | None = None,
    linear_3: LogicalTensor | None = None,
    unsqueeze_3: LogicalTensor | None = None,
    getitem_6: LogicalTensor | None = None,
    getitem_7: LogicalTensor | None = None,
    getitem_8: LogicalTensor | None = None,
    getitem_9: LogicalTensor | None = None,
    getitem_10: LogicalTensor | None = None,
    getitem_11: LogicalTensor | None = None,
    silu_3: LogicalTensor | None = None,
    linear_4: LogicalTensor | None = None,
    unsqueeze_4: LogicalTensor | None = None,
    getitem_12: LogicalTensor | None = None,
    getitem_13: LogicalTensor | None = None,
    getitem_14: LogicalTensor | None = None,
    linear_5: LogicalTensor | None = None,
    linear_6: LogicalTensor | None = None,
    select: LogicalTensor | None = None,
    arange_1: LogicalTensor | None = None,
    div_1: LogicalTensor | None = None,
    pow_1: LogicalTensor | None = None,
    reciprocal: LogicalTensor | None = None,
    mul_3: LogicalTensor | None = None,
    einsum: LogicalTensor | None = None,
    cos_1: LogicalTensor | None = None,
    sin_1: LogicalTensor | None = None,
    neg: LogicalTensor | None = None,
    sin_2: LogicalTensor | None = None,
    cos_2: LogicalTensor | None = None,
    stack: LogicalTensor | None = None,
    reshape: LogicalTensor | None = None,
    to_2: LogicalTensor | None = None,
    select_1: LogicalTensor | None = None,
    arange_2: LogicalTensor | None = None,
    div_2: LogicalTensor | None = None,
    pow_2: LogicalTensor | None = None,
    reciprocal_1: LogicalTensor | None = None,
    mul_4: LogicalTensor | None = None,
    einsum_1: LogicalTensor | None = None,
    cos_3: LogicalTensor | None = None,
    sin_3: LogicalTensor | None = None,
    neg_1: LogicalTensor | None = None,
    sin_4: LogicalTensor | None = None,
    cos_4: LogicalTensor | None = None,
    stack_1: LogicalTensor | None = None,
    reshape_1: LogicalTensor | None = None,
    to_3: LogicalTensor | None = None,
    select_2: LogicalTensor | None = None,
    arange_3: LogicalTensor | None = None,
    div_3: LogicalTensor | None = None,
    pow_3: LogicalTensor | None = None,
    reciprocal_2: LogicalTensor | None = None,
    mul_5: LogicalTensor | None = None,
    einsum_2: LogicalTensor | None = None,
    cos_5: LogicalTensor | None = None,
    sin_5: LogicalTensor | None = None,
    neg_2: LogicalTensor | None = None,
    sin_6: LogicalTensor | None = None,
    cos_6: LogicalTensor | None = None,
    stack_2: LogicalTensor | None = None,
    reshape_2: LogicalTensor | None = None,
    to_4: LogicalTensor | None = None,
    select_3: LogicalTensor | None = None,
    arange_4: LogicalTensor | None = None,
    div_4: LogicalTensor | None = None,
    pow_4: LogicalTensor | None = None,
    reciprocal_3: LogicalTensor | None = None,
    mul_6: LogicalTensor | None = None,
    einsum_3: LogicalTensor | None = None,
    cos_7: LogicalTensor | None = None,
    sin_7: LogicalTensor | None = None,
    neg_3: LogicalTensor | None = None,
    sin_8: LogicalTensor | None = None,
    cos_8: LogicalTensor | None = None,
    stack_3: LogicalTensor | None = None,
    reshape_3: LogicalTensor | None = None,
    to_5: LogicalTensor | None = None,
    cat_1: LogicalTensor | None = None,
    unsqueeze_5: LogicalTensor | None = None,
    select_4: LogicalTensor | None = None,
    arange_5: LogicalTensor | None = None,
    div_5: LogicalTensor | None = None,
    pow_5: LogicalTensor | None = None,
    reciprocal_4: LogicalTensor | None = None,
    mul_7: LogicalTensor | None = None,
    einsum_4: LogicalTensor | None = None,
    cos_9: LogicalTensor | None = None,
    sin_9: LogicalTensor | None = None,
    neg_4: LogicalTensor | None = None,
    sin_10: LogicalTensor | None = None,
    cos_10: LogicalTensor | None = None,
    stack_4: LogicalTensor | None = None,
    reshape_4: LogicalTensor | None = None,
    to_6: LogicalTensor | None = None,
    select_5: LogicalTensor | None = None,
    arange_6: LogicalTensor | None = None,
    div_6: LogicalTensor | None = None,
    pow_6: LogicalTensor | None = None,
    reciprocal_5: LogicalTensor | None = None,
    mul_8: LogicalTensor | None = None,
    einsum_5: LogicalTensor | None = None,
    cos_11: LogicalTensor | None = None,
    sin_11: LogicalTensor | None = None,
    neg_5: LogicalTensor | None = None,
    sin_12: LogicalTensor | None = None,
    cos_12: LogicalTensor | None = None,
    stack_5: LogicalTensor | None = None,
    reshape_5: LogicalTensor | None = None,
    to_7: LogicalTensor | None = None,
    select_6: LogicalTensor | None = None,
    arange_7: LogicalTensor | None = None,
    div_7: LogicalTensor | None = None,
    pow_7: LogicalTensor | None = None,
    reciprocal_6: LogicalTensor | None = None,
    mul_9: LogicalTensor | None = None,
    einsum_6: LogicalTensor | None = None,
    cos_13: LogicalTensor | None = None,
    sin_13: LogicalTensor | None = None,
    neg_6: LogicalTensor | None = None,
    sin_14: LogicalTensor | None = None,
    cos_14: LogicalTensor | None = None,
    stack_6: LogicalTensor | None = None,
    reshape_6: LogicalTensor | None = None,
    to_8: LogicalTensor | None = None,
    select_7: LogicalTensor | None = None,
    arange_8: LogicalTensor | None = None,
    div_8: LogicalTensor | None = None,
    pow_8: LogicalTensor | None = None,
    reciprocal_7: LogicalTensor | None = None,
    mul_10: LogicalTensor | None = None,
    einsum_7: LogicalTensor | None = None,
    cos_15: LogicalTensor | None = None,
    sin_15: LogicalTensor | None = None,
    neg_7: LogicalTensor | None = None,
    sin_16: LogicalTensor | None = None,
    cos_16: LogicalTensor | None = None,
    stack_7: LogicalTensor | None = None,
    reshape_7: LogicalTensor | None = None,
    to_9: LogicalTensor | None = None,
    cat_2: LogicalTensor | None = None,
    unsqueeze_6: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> FluxPrologueTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('linear_5', 'linear_6', 'unsqueeze_5', 'unsqueeze_6', 'linear_1', 'getitem', 'getitem_1', 'getitem_2', 'getitem_3', 'getitem_4', 'getitem_5', 'getitem_6', 'getitem_7', 'getitem_8', 'getitem_9', 'getitem_10', 'getitem_11', 'getitem_12', 'getitem_13', 'getitem_14')))
    tensors = FluxPrologueTensors(
        p_img_in_weight=_bind_tensor(
            p_img_in_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key="img_in.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec("img_in.weight", dtype='float32', shape=(4096, 128)),
                layout=_quantized_weight_layout("img_in.weight", dtype='float32', shape=(4096, 128)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_img_in_weight' in request_state_outputs,
            ),
        ),
        p_time_in_in_layer_weight=_bind_tensor(
            p_time_in_in_layer_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key="time_in.in_layer.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec("time_in.in_layer.weight", dtype='float32', shape=(4096, 256)),
                layout=_quantized_weight_layout("time_in.in_layer.weight", dtype='float32', shape=(4096, 256)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_time_in_in_layer_weight' in request_state_outputs,
            ),
        ),
        p_time_in_out_layer_weight=_bind_tensor(
            p_time_in_out_layer_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key="time_in.out_layer.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec("time_in.out_layer.weight", dtype='float32', shape=(4096, 4096)),
                layout=_quantized_weight_layout("time_in.out_layer.weight", dtype='float32', shape=(4096, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_time_in_out_layer_weight' in request_state_outputs,
            ),
        ),
        p_txt_in_weight=_bind_tensor(
            p_txt_in_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key="txt_in.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec("txt_in.weight", dtype='float32', shape=(4096, 12288)),
                layout=_quantized_weight_layout("txt_in.weight", dtype='float32', shape=(4096, 12288)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_txt_in_weight' in request_state_outputs,
            ),
        ),
        p_double_stream_modulation_img_lin_weight=_bind_tensor(
            p_double_stream_modulation_img_lin_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key="double_stream_modulation_img.lin.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec("double_stream_modulation_img.lin.weight", dtype='float32', shape=(24576, 4096)),
                layout=_quantized_weight_layout("double_stream_modulation_img.lin.weight", dtype='float32', shape=(24576, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_double_stream_modulation_img_lin_weight' in request_state_outputs,
            ),
        ),
        p_double_stream_modulation_txt_lin_weight=_bind_tensor(
            p_double_stream_modulation_txt_lin_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key="double_stream_modulation_txt.lin.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec("double_stream_modulation_txt.lin.weight", dtype='float32', shape=(24576, 4096)),
                layout=_quantized_weight_layout("double_stream_modulation_txt.lin.weight", dtype='float32', shape=(24576, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_double_stream_modulation_txt_lin_weight' in request_state_outputs,
            ),
        ),
        p_single_stream_modulation_lin_weight=_bind_tensor(
            p_single_stream_modulation_lin_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key="single_stream_modulation.lin.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec("single_stream_modulation.lin.weight", dtype='float32', shape=(12288, 4096)),
                layout=_quantized_weight_layout("single_stream_modulation.lin.weight", dtype='float32', shape=(12288, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_single_stream_modulation_lin_weight' in request_state_outputs,
            ),
        ),
        x=_bind_tensor(
            x,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='x' in request_state_outputs,
            ),
        ),
        x_ids=_bind_tensor(
            x_ids,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(1, image_seq_len, 4)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='x_ids' in request_state_outputs,
            ),
        ),
        timesteps=_bind_tensor(
            timesteps,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='timesteps' in request_state_outputs,
            ),
        ),
        ctx=_bind_tensor(
            ctx,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='ctx' in request_state_outputs,
            ),
        ),
        ctx_ids=_bind_tensor(
            ctx_ids,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(1, text_seq_len, 4)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='ctx_ids' in request_state_outputs,
            ),
        ),
        mul=_bind_tensor(
            mul,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul' in request_state_outputs,
            ),
        ),
        arange=_bind_tensor(
            arange,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='arange',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(128,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='arange' in request_state_outputs,
            ),
        ),
        mul_1=_bind_tensor(
            mul_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(128,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_1' in request_state_outputs,
            ),
        ),
        div=_bind_tensor(
            div,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='div',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(128,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='div' in request_state_outputs,
            ),
        ),
        exp=_bind_tensor(
            exp,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='exp',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(128,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='exp' in request_state_outputs,
            ),
        ),
        unsqueeze=_bind_tensor(
            unsqueeze,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze' in request_state_outputs,
            ),
        ),
        to=_bind_tensor(
            to,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to' in request_state_outputs,
            ),
        ),
        unsqueeze_1=_bind_tensor(
            unsqueeze_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze_1' in request_state_outputs,
            ),
        ),
        mul_2=_bind_tensor(
            mul_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_2' in request_state_outputs,
            ),
        ),
        cos=_bind_tensor(
            cos,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos' in request_state_outputs,
            ),
        ),
        sin=_bind_tensor(
            sin,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin' in request_state_outputs,
            ),
        ),
        cat=_bind_tensor(
            cat,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat' in request_state_outputs,
            ),
        ),
        to_1=_bind_tensor(
            to_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 256)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_1' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            linear,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear' in request_state_outputs,
            ),
        ),
        silu=_bind_tensor(
            silu,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='silu',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='silu' in request_state_outputs,
            ),
        ),
        linear_1=_bind_tensor(
            linear_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_1' in request_state_outputs,
            ),
        ),
        silu_1=_bind_tensor(
            silu_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='silu_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='silu_1' in request_state_outputs,
            ),
        ),
        linear_2=_bind_tensor(
            linear_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 24576)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_2' in request_state_outputs,
            ),
        ),
        unsqueeze_2=_bind_tensor(
            unsqueeze_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 24576)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze_2' in request_state_outputs,
            ),
        ),
        getitem=_bind_tensor(
            getitem,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem' in request_state_outputs,
            ),
        ),
        getitem_1=_bind_tensor(
            getitem_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_1' in request_state_outputs,
            ),
        ),
        getitem_2=_bind_tensor(
            getitem_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_2' in request_state_outputs,
            ),
        ),
        getitem_3=_bind_tensor(
            getitem_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_3' in request_state_outputs,
            ),
        ),
        getitem_4=_bind_tensor(
            getitem_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_4' in request_state_outputs,
            ),
        ),
        getitem_5=_bind_tensor(
            getitem_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_5' in request_state_outputs,
            ),
        ),
        silu_2=_bind_tensor(
            silu_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='silu_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='silu_2' in request_state_outputs,
            ),
        ),
        linear_3=_bind_tensor(
            linear_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 24576)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_3' in request_state_outputs,
            ),
        ),
        unsqueeze_3=_bind_tensor(
            unsqueeze_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 24576)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze_3' in request_state_outputs,
            ),
        ),
        getitem_6=_bind_tensor(
            getitem_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_6' in request_state_outputs,
            ),
        ),
        getitem_7=_bind_tensor(
            getitem_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_7' in request_state_outputs,
            ),
        ),
        getitem_8=_bind_tensor(
            getitem_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_8',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_8' in request_state_outputs,
            ),
        ),
        getitem_9=_bind_tensor(
            getitem_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_9',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_9' in request_state_outputs,
            ),
        ),
        getitem_10=_bind_tensor(
            getitem_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_10',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_10' in request_state_outputs,
            ),
        ),
        getitem_11=_bind_tensor(
            getitem_11,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_11',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_11' in request_state_outputs,
            ),
        ),
        silu_3=_bind_tensor(
            silu_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='silu_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='silu_3' in request_state_outputs,
            ),
        ),
        linear_4=_bind_tensor(
            linear_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_4' in request_state_outputs,
            ),
        ),
        unsqueeze_4=_bind_tensor(
            unsqueeze_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze_4' in request_state_outputs,
            ),
        ),
        getitem_12=_bind_tensor(
            getitem_12,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_12',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_12' in request_state_outputs,
            ),
        ),
        getitem_13=_bind_tensor(
            getitem_13,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_13',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_13' in request_state_outputs,
            ),
        ),
        getitem_14=_bind_tensor(
            getitem_14,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_14',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_14' in request_state_outputs,
            ),
        ),
        linear_5=_bind_tensor(
            linear_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_5' in request_state_outputs,
            ),
        ),
        linear_6=_bind_tensor(
            linear_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_6' in request_state_outputs,
            ),
        ),
        select=_bind_tensor(
            select,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(1, image_seq_len)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select' in request_state_outputs,
            ),
        ),
        arange_1=_bind_tensor(
            arange_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='arange_1',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='arange_1' in request_state_outputs,
            ),
        ),
        div_1=_bind_tensor(
            div_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='div_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='div_1' in request_state_outputs,
            ),
        ),
        pow_1=_bind_tensor(
            pow_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='pow_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_1' in request_state_outputs,
            ),
        ),
        reciprocal=_bind_tensor(
            reciprocal,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reciprocal',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reciprocal' in request_state_outputs,
            ),
        ),
        mul_3=_bind_tensor(
            mul_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_3' in request_state_outputs,
            ),
        ),
        einsum=_bind_tensor(
            einsum,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='einsum',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='einsum' in request_state_outputs,
            ),
        ),
        cos_1=_bind_tensor(
            cos_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_1' in request_state_outputs,
            ),
        ),
        sin_1=_bind_tensor(
            sin_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_1' in request_state_outputs,
            ),
        ),
        neg=_bind_tensor(
            neg,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='neg' in request_state_outputs,
            ),
        ),
        sin_2=_bind_tensor(
            sin_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_2' in request_state_outputs,
            ),
        ),
        cos_2=_bind_tensor(
            cos_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_2' in request_state_outputs,
            ),
        ),
        stack=_bind_tensor(
            stack,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='stack',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 4)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='stack' in request_state_outputs,
            ),
        ),
        reshape=_bind_tensor(
            reshape,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape' in request_state_outputs,
            ),
        ),
        to_2=_bind_tensor(
            to_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_2' in request_state_outputs,
            ),
        ),
        select_1=_bind_tensor(
            select_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_1',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(1, image_seq_len)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_1' in request_state_outputs,
            ),
        ),
        arange_2=_bind_tensor(
            arange_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='arange_2',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='arange_2' in request_state_outputs,
            ),
        ),
        div_2=_bind_tensor(
            div_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='div_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='div_2' in request_state_outputs,
            ),
        ),
        pow_2=_bind_tensor(
            pow_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='pow_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_2' in request_state_outputs,
            ),
        ),
        reciprocal_1=_bind_tensor(
            reciprocal_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reciprocal_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reciprocal_1' in request_state_outputs,
            ),
        ),
        mul_4=_bind_tensor(
            mul_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_4' in request_state_outputs,
            ),
        ),
        einsum_1=_bind_tensor(
            einsum_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='einsum_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='einsum_1' in request_state_outputs,
            ),
        ),
        cos_3=_bind_tensor(
            cos_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_3' in request_state_outputs,
            ),
        ),
        sin_3=_bind_tensor(
            sin_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_3' in request_state_outputs,
            ),
        ),
        neg_1=_bind_tensor(
            neg_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='neg_1' in request_state_outputs,
            ),
        ),
        sin_4=_bind_tensor(
            sin_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_4' in request_state_outputs,
            ),
        ),
        cos_4=_bind_tensor(
            cos_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_4' in request_state_outputs,
            ),
        ),
        stack_1=_bind_tensor(
            stack_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='stack_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 4)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='stack_1' in request_state_outputs,
            ),
        ),
        reshape_1=_bind_tensor(
            reshape_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_1' in request_state_outputs,
            ),
        ),
        to_3=_bind_tensor(
            to_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_3' in request_state_outputs,
            ),
        ),
        select_2=_bind_tensor(
            select_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_2',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(1, image_seq_len)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_2' in request_state_outputs,
            ),
        ),
        arange_3=_bind_tensor(
            arange_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='arange_3',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='arange_3' in request_state_outputs,
            ),
        ),
        div_3=_bind_tensor(
            div_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='div_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='div_3' in request_state_outputs,
            ),
        ),
        pow_3=_bind_tensor(
            pow_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='pow_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_3' in request_state_outputs,
            ),
        ),
        reciprocal_2=_bind_tensor(
            reciprocal_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reciprocal_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reciprocal_2' in request_state_outputs,
            ),
        ),
        mul_5=_bind_tensor(
            mul_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_5' in request_state_outputs,
            ),
        ),
        einsum_2=_bind_tensor(
            einsum_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='einsum_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='einsum_2' in request_state_outputs,
            ),
        ),
        cos_5=_bind_tensor(
            cos_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_5' in request_state_outputs,
            ),
        ),
        sin_5=_bind_tensor(
            sin_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_5' in request_state_outputs,
            ),
        ),
        neg_2=_bind_tensor(
            neg_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='neg_2' in request_state_outputs,
            ),
        ),
        sin_6=_bind_tensor(
            sin_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_6' in request_state_outputs,
            ),
        ),
        cos_6=_bind_tensor(
            cos_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_6' in request_state_outputs,
            ),
        ),
        stack_2=_bind_tensor(
            stack_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='stack_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 4)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='stack_2' in request_state_outputs,
            ),
        ),
        reshape_2=_bind_tensor(
            reshape_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_2' in request_state_outputs,
            ),
        ),
        to_4=_bind_tensor(
            to_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_4' in request_state_outputs,
            ),
        ),
        select_3=_bind_tensor(
            select_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_3',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(1, image_seq_len)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_3' in request_state_outputs,
            ),
        ),
        arange_4=_bind_tensor(
            arange_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='arange_4',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='arange_4' in request_state_outputs,
            ),
        ),
        div_4=_bind_tensor(
            div_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='div_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='div_4' in request_state_outputs,
            ),
        ),
        pow_4=_bind_tensor(
            pow_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='pow_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_4' in request_state_outputs,
            ),
        ),
        reciprocal_3=_bind_tensor(
            reciprocal_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reciprocal_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reciprocal_3' in request_state_outputs,
            ),
        ),
        mul_6=_bind_tensor(
            mul_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_6' in request_state_outputs,
            ),
        ),
        einsum_3=_bind_tensor(
            einsum_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='einsum_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='einsum_3' in request_state_outputs,
            ),
        ),
        cos_7=_bind_tensor(
            cos_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_7' in request_state_outputs,
            ),
        ),
        sin_7=_bind_tensor(
            sin_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_7' in request_state_outputs,
            ),
        ),
        neg_3=_bind_tensor(
            neg_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='neg_3' in request_state_outputs,
            ),
        ),
        sin_8=_bind_tensor(
            sin_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_8',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_8' in request_state_outputs,
            ),
        ),
        cos_8=_bind_tensor(
            cos_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_8',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_8' in request_state_outputs,
            ),
        ),
        stack_3=_bind_tensor(
            stack_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='stack_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 4)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='stack_3' in request_state_outputs,
            ),
        ),
        reshape_3=_bind_tensor(
            reshape_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_3' in request_state_outputs,
            ),
        ),
        to_5=_bind_tensor(
            to_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_5' in request_state_outputs,
            ),
        ),
        cat_1=_bind_tensor(
            cat_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 64, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat_1' in request_state_outputs,
            ),
        ),
        unsqueeze_5=_bind_tensor(
            unsqueeze_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, image_seq_len, 64, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze_5' in request_state_outputs,
            ),
        ),
        select_4=_bind_tensor(
            select_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_4',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(1, text_seq_len)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_4' in request_state_outputs,
            ),
        ),
        arange_5=_bind_tensor(
            arange_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='arange_5',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='arange_5' in request_state_outputs,
            ),
        ),
        div_5=_bind_tensor(
            div_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='div_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='div_5' in request_state_outputs,
            ),
        ),
        pow_5=_bind_tensor(
            pow_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='pow_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_5' in request_state_outputs,
            ),
        ),
        reciprocal_4=_bind_tensor(
            reciprocal_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reciprocal_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reciprocal_4' in request_state_outputs,
            ),
        ),
        mul_7=_bind_tensor(
            mul_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_7' in request_state_outputs,
            ),
        ),
        einsum_4=_bind_tensor(
            einsum_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='einsum_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='einsum_4' in request_state_outputs,
            ),
        ),
        cos_9=_bind_tensor(
            cos_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_9',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_9' in request_state_outputs,
            ),
        ),
        sin_9=_bind_tensor(
            sin_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_9',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_9' in request_state_outputs,
            ),
        ),
        neg_4=_bind_tensor(
            neg_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='neg_4' in request_state_outputs,
            ),
        ),
        sin_10=_bind_tensor(
            sin_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_10',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_10' in request_state_outputs,
            ),
        ),
        cos_10=_bind_tensor(
            cos_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_10',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_10' in request_state_outputs,
            ),
        ),
        stack_4=_bind_tensor(
            stack_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='stack_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 4)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='stack_4' in request_state_outputs,
            ),
        ),
        reshape_4=_bind_tensor(
            reshape_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_4' in request_state_outputs,
            ),
        ),
        to_6=_bind_tensor(
            to_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_6' in request_state_outputs,
            ),
        ),
        select_5=_bind_tensor(
            select_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_5',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(1, text_seq_len)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_5' in request_state_outputs,
            ),
        ),
        arange_6=_bind_tensor(
            arange_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='arange_6',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='arange_6' in request_state_outputs,
            ),
        ),
        div_6=_bind_tensor(
            div_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='div_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='div_6' in request_state_outputs,
            ),
        ),
        pow_6=_bind_tensor(
            pow_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='pow_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_6' in request_state_outputs,
            ),
        ),
        reciprocal_5=_bind_tensor(
            reciprocal_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reciprocal_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reciprocal_5' in request_state_outputs,
            ),
        ),
        mul_8=_bind_tensor(
            mul_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_8',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_8' in request_state_outputs,
            ),
        ),
        einsum_5=_bind_tensor(
            einsum_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='einsum_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='einsum_5' in request_state_outputs,
            ),
        ),
        cos_11=_bind_tensor(
            cos_11,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_11',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_11' in request_state_outputs,
            ),
        ),
        sin_11=_bind_tensor(
            sin_11,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_11',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_11' in request_state_outputs,
            ),
        ),
        neg_5=_bind_tensor(
            neg_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='neg_5' in request_state_outputs,
            ),
        ),
        sin_12=_bind_tensor(
            sin_12,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_12',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_12' in request_state_outputs,
            ),
        ),
        cos_12=_bind_tensor(
            cos_12,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_12',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_12' in request_state_outputs,
            ),
        ),
        stack_5=_bind_tensor(
            stack_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='stack_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 4)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='stack_5' in request_state_outputs,
            ),
        ),
        reshape_5=_bind_tensor(
            reshape_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_5' in request_state_outputs,
            ),
        ),
        to_7=_bind_tensor(
            to_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_7' in request_state_outputs,
            ),
        ),
        select_6=_bind_tensor(
            select_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_6',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(1, text_seq_len)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_6' in request_state_outputs,
            ),
        ),
        arange_7=_bind_tensor(
            arange_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='arange_7',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='arange_7' in request_state_outputs,
            ),
        ),
        div_7=_bind_tensor(
            div_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='div_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='div_7' in request_state_outputs,
            ),
        ),
        pow_7=_bind_tensor(
            pow_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='pow_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_7' in request_state_outputs,
            ),
        ),
        reciprocal_6=_bind_tensor(
            reciprocal_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reciprocal_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reciprocal_6' in request_state_outputs,
            ),
        ),
        mul_9=_bind_tensor(
            mul_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_9',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_9' in request_state_outputs,
            ),
        ),
        einsum_6=_bind_tensor(
            einsum_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='einsum_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='einsum_6' in request_state_outputs,
            ),
        ),
        cos_13=_bind_tensor(
            cos_13,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_13',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_13' in request_state_outputs,
            ),
        ),
        sin_13=_bind_tensor(
            sin_13,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_13',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_13' in request_state_outputs,
            ),
        ),
        neg_6=_bind_tensor(
            neg_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='neg_6' in request_state_outputs,
            ),
        ),
        sin_14=_bind_tensor(
            sin_14,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_14',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_14' in request_state_outputs,
            ),
        ),
        cos_14=_bind_tensor(
            cos_14,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_14',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_14' in request_state_outputs,
            ),
        ),
        stack_6=_bind_tensor(
            stack_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='stack_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 4)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='stack_6' in request_state_outputs,
            ),
        ),
        reshape_6=_bind_tensor(
            reshape_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_6' in request_state_outputs,
            ),
        ),
        to_8=_bind_tensor(
            to_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_8',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_8' in request_state_outputs,
            ),
        ),
        select_7=_bind_tensor(
            select_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_7',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(1, text_seq_len)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_7' in request_state_outputs,
            ),
        ),
        arange_8=_bind_tensor(
            arange_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='arange_8',
                layer=prefix,
                spec=TensorSpec(dtype='int64', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='arange_8' in request_state_outputs,
            ),
        ),
        div_8=_bind_tensor(
            div_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='div_8',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='div_8' in request_state_outputs,
            ),
        ),
        pow_8=_bind_tensor(
            pow_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='pow_8',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_8' in request_state_outputs,
            ),
        ),
        reciprocal_7=_bind_tensor(
            reciprocal_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reciprocal_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reciprocal_7' in request_state_outputs,
            ),
        ),
        mul_10=_bind_tensor(
            mul_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_10',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(16,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_10' in request_state_outputs,
            ),
        ),
        einsum_7=_bind_tensor(
            einsum_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='einsum_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='einsum_7' in request_state_outputs,
            ),
        ),
        cos_15=_bind_tensor(
            cos_15,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_15',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_15' in request_state_outputs,
            ),
        ),
        sin_15=_bind_tensor(
            sin_15,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_15',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_15' in request_state_outputs,
            ),
        ),
        neg_7=_bind_tensor(
            neg_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='neg_7' in request_state_outputs,
            ),
        ),
        sin_16=_bind_tensor(
            sin_16,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='sin_16',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='sin_16' in request_state_outputs,
            ),
        ),
        cos_16=_bind_tensor(
            cos_16,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cos_16',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cos_16' in request_state_outputs,
            ),
        ),
        stack_7=_bind_tensor(
            stack_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='stack_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 4)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='stack_7' in request_state_outputs,
            ),
        ),
        reshape_7=_bind_tensor(
            reshape_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_7' in request_state_outputs,
            ),
        ),
        to_9=_bind_tensor(
            to_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_9',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 16, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_9' in request_state_outputs,
            ),
        ),
        cat_2=_bind_tensor(
            cat_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len, 64, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat_2' in request_state_outputs,
            ),
        ),
        unsqueeze_6=_bind_tensor(
            unsqueeze_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len, 64, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze_6' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    _bind_alias_source(tensors.mul, tensors.unsqueeze)
    _bind_alias_source(tensors.unsqueeze, tensors.to)
    _bind_alias_source(tensors.exp, tensors.unsqueeze_1)
    _bind_alias_source(tensors.cat, tensors.to_1)
    _bind_alias_source(tensors.linear_2, tensors.unsqueeze_2)
    _bind_alias_source(tensors.linear_3, tensors.unsqueeze_3)
    _bind_alias_source(tensors.linear_4, tensors.unsqueeze_4)
    _bind_alias_source(tensors.stack, tensors.reshape)
    _bind_alias_source(tensors.reshape, tensors.to_2)
    _bind_alias_source(tensors.stack_1, tensors.reshape_1)
    _bind_alias_source(tensors.reshape_1, tensors.to_3)
    _bind_alias_source(tensors.stack_2, tensors.reshape_2)
    _bind_alias_source(tensors.reshape_2, tensors.to_4)
    _bind_alias_source(tensors.stack_3, tensors.reshape_3)
    _bind_alias_source(tensors.reshape_3, tensors.to_5)
    _bind_alias_source(tensors.cat_1, tensors.unsqueeze_5)
    _bind_alias_source(tensors.stack_4, tensors.reshape_4)
    _bind_alias_source(tensors.reshape_4, tensors.to_6)
    _bind_alias_source(tensors.stack_5, tensors.reshape_5)
    _bind_alias_source(tensors.reshape_5, tensors.to_7)
    _bind_alias_source(tensors.stack_6, tensors.reshape_6)
    _bind_alias_source(tensors.reshape_6, tensors.to_8)
    _bind_alias_source(tensors.stack_7, tensors.reshape_7)
    _bind_alias_source(tensors.reshape_7, tensors.to_9)
    _bind_alias_source(tensors.cat_2, tensors.unsqueeze_6)
    return tensors


_F16_TENSOR_NAMES = frozenset(())
_F16_TENSOR_PREFIXES = ()
_Q6_TENSOR_NAMES = frozenset(())
_Q6_TENSOR_PREFIXES = ()
_Q8_TENSOR_NAMES = frozenset(())
_Q8_TENSOR_PREFIXES = ('',)


def _quantized_weight_spec(checkpoint_key: str, *, dtype: str, shape: tuple[int, ...]) -> TensorSpec:
    force_f16 = checkpoint_key in _F16_TENSOR_NAMES or checkpoint_key.startswith(_F16_TENSOR_PREFIXES)
    if force_f16:
        return TensorSpec(dtype="float16", shape=shape)
    if dtype not in ("float32", "float16", "bfloat16"):
        return TensorSpec(dtype=dtype, shape=shape)
    force_q8 = checkpoint_key in _Q8_TENSOR_NAMES or checkpoint_key.startswith(_Q8_TENSOR_PREFIXES)
    force_q6 = checkpoint_key in _Q6_TENSOR_NAMES or checkpoint_key.startswith(_Q6_TENSOR_PREFIXES)
    if force_q8 and len(shape) >= 2:
        n, k = _quantized_matrix_shape(shape)
        padded_k = _round_up(k, 32)
        return TensorSpec(dtype="uint16", shape=(n, padded_k // 32 * 17))
    if force_q6 and len(shape) >= 2:
        n, k = _quantized_matrix_shape(shape)
        if k % 256 != 0:
            raise ValueError(f"Q6_K tensor {checkpoint_key} requires K to be divisible by 256, got {k}")
        return TensorSpec(dtype="uint16", shape=(n, k // 256 * 105))
    if len(shape) != 2:
        return TensorSpec(dtype=dtype, shape=shape)
    n, k = shape
    if k % 256 != 0:
        if k % 32 != 0:
            return TensorSpec(dtype="float32", shape=shape)
        return TensorSpec(dtype="uint16", shape=(n, k // 32 * 17))
    return TensorSpec(dtype="uint32", shape=(n, k // 256 * 36))


def _quantized_weight_layout(checkpoint_key: str, *, dtype: str, shape: tuple[int, ...]) -> TensorLayout:
    force_f16 = checkpoint_key in _F16_TENSOR_NAMES or checkpoint_key.startswith(_F16_TENSOR_PREFIXES)
    if force_f16:
        return CONTIGUOUS_LAYOUT
    if dtype not in ("float32", "float16", "bfloat16"):
        return CONTIGUOUS_LAYOUT
    force_q8 = checkpoint_key in _Q8_TENSOR_NAMES or checkpoint_key.startswith(_Q8_TENSOR_PREFIXES)
    force_q6 = checkpoint_key in _Q6_TENSOR_NAMES or checkpoint_key.startswith(_Q6_TENSOR_PREFIXES)
    if force_q8 and len(shape) >= 2:
        _, k = _quantized_matrix_shape(shape)
        return q8_0_halfwords_layout(logical_k=k)
    if force_q6 and len(shape) >= 2:
        _, k = _quantized_matrix_shape(shape)
        if k % 256 != 0:
            raise ValueError(f"Q6_K tensor {checkpoint_key} requires K to be divisible by 256, got {k}")
        return q6_k_halfwords_layout(logical_k=k)
    if len(shape) != 2:
        return CONTIGUOUS_LAYOUT
    _, k = shape
    if k % 256 != 0:
        if k % 32 != 0:
            return CONTIGUOUS_LAYOUT
        return q8_0_halfwords_layout(logical_k=k)
    return q4_k_words_layout(logical_k=k)


def _quantized_matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    rows = shape[0]
    cols = 1
    for dim in shape[1:]:
        cols *= dim
    return rows, cols


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _declare_tensor(
    *,
    spec: TensorSpec,
    role: TensorRole,
    memory: MemoryClass,
    lifetime: TensorLifetime,
    layout: TensorLayout = CONTIGUOUS_LAYOUT,
    checkpoint: str | None = None,
    checkpoint_key: str | None = None,
    reference_key: str | None = None,
    layer: str | None = None,
    request_state: bool = False,
) -> LogicalTensor:
    if request_state:
        role = TensorRole.OUTPUT
        memory = MemoryClass.REQUEST_STATE
        lifetime = TensorLifetime.REQUEST
    return LogicalTensor(
        spec=spec,
        role=role,
        memory=memory,
        lifetime=lifetime,
        checkpoint=checkpoint,
        checkpoint_key=checkpoint_key,
        reference_key=reference_key,
        layer=layer,
        layout=layout,
    )


def _bind_tensor(
    bound: LogicalTensor | None,
    tensor: LogicalTensor,
) -> LogicalTensor:
    if bound is None:
        return tensor
    if bound.spec != tensor.spec:
        bound_name = bound.name or "<bound>"
        tensor_name = tensor.name or "<declared>"
        raise ValueError(f"{bound_name} spec {bound.spec} does not match {tensor_name} spec {tensor.spec}")
    return bound


def _bind_alias_source(src: LogicalTensor, dst: LogicalTensor) -> None:
    bind_logical_tensor_alias(src, dst)


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
