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
class FluxSingleBlockTensors:
    p_linear1_weight: LogicalTensor
    p_linear2_weight: LogicalTensor
    p_norm_query_norm_scale: LogicalTensor
    p_norm_key_norm_scale: LogicalTensor
    hidden_states: LogicalTensor
    pe: LogicalTensor
    mod_shift: LogicalTensor
    mod_scale: LogicalTensor
    mod_gate: LogicalTensor
    add: LogicalTensor
    layer_norm: LogicalTensor
    mul: LogicalTensor
    add_1: LogicalTensor
    linear: LogicalTensor
    getitem: LogicalTensor
    getitem_1: LogicalTensor
    reshape: LogicalTensor
    permute: LogicalTensor
    getitem_2: LogicalTensor
    getitem_3: LogicalTensor
    getitem_4: LogicalTensor
    to: LogicalTensor
    pow_1: LogicalTensor
    mean: LogicalTensor
    add_2: LogicalTensor
    rsqrt: LogicalTensor
    mul_1: LogicalTensor
    to_1: LogicalTensor
    mul_2: LogicalTensor
    to_2: LogicalTensor
    pow_2: LogicalTensor
    mean_1: LogicalTensor
    add_3: LogicalTensor
    rsqrt_1: LogicalTensor
    mul_3: LogicalTensor
    to_3: LogicalTensor
    mul_4: LogicalTensor
    to_4: LogicalTensor
    to_5: LogicalTensor
    to_6: LogicalTensor
    reshape_1: LogicalTensor
    to_7: LogicalTensor
    reshape_2: LogicalTensor
    select: LogicalTensor
    select_1: LogicalTensor
    mul_5: LogicalTensor
    select_2: LogicalTensor
    select_3: LogicalTensor
    mul_6: LogicalTensor
    add_4: LogicalTensor
    select_4: LogicalTensor
    select_5: LogicalTensor
    mul_7: LogicalTensor
    select_6: LogicalTensor
    select_7: LogicalTensor
    mul_8: LogicalTensor
    add_5: LogicalTensor
    reshape_3: LogicalTensor
    type_as: LogicalTensor
    reshape_4: LogicalTensor
    type_as_1: LogicalTensor
    scaled_dot_product_attention: LogicalTensor
    permute_1: LogicalTensor
    reshape_5: LogicalTensor
    getitem_5: LogicalTensor
    getitem_6: LogicalTensor
    silu: LogicalTensor
    mul_9: LogicalTensor
    cat: LogicalTensor
    linear_1: LogicalTensor
    mul_10: LogicalTensor
    add_6: LogicalTensor


FLUX_SINGLE_BLOCK_OUTPUT: str = 'add_6'


def create_flux_single_block(
    prefix: str,
    layer_idx: int,
    *,
    text_seq_len: int,
    image_seq_len: int,
    p_linear1_weight: LogicalTensor | None = None,
    p_linear2_weight: LogicalTensor | None = None,
    p_norm_query_norm_scale: LogicalTensor | None = None,
    p_norm_key_norm_scale: LogicalTensor | None = None,
    hidden_states: LogicalTensor | None = None,
    pe: LogicalTensor | None = None,
    mod_shift: LogicalTensor | None = None,
    mod_scale: LogicalTensor | None = None,
    mod_gate: LogicalTensor | None = None,
    add: LogicalTensor | None = None,
    layer_norm: LogicalTensor | None = None,
    mul: LogicalTensor | None = None,
    add_1: LogicalTensor | None = None,
    linear: LogicalTensor | None = None,
    getitem: LogicalTensor | None = None,
    getitem_1: LogicalTensor | None = None,
    reshape: LogicalTensor | None = None,
    permute: LogicalTensor | None = None,
    getitem_2: LogicalTensor | None = None,
    getitem_3: LogicalTensor | None = None,
    getitem_4: LogicalTensor | None = None,
    to: LogicalTensor | None = None,
    pow_1: LogicalTensor | None = None,
    mean: LogicalTensor | None = None,
    add_2: LogicalTensor | None = None,
    rsqrt: LogicalTensor | None = None,
    mul_1: LogicalTensor | None = None,
    to_1: LogicalTensor | None = None,
    mul_2: LogicalTensor | None = None,
    to_2: LogicalTensor | None = None,
    pow_2: LogicalTensor | None = None,
    mean_1: LogicalTensor | None = None,
    add_3: LogicalTensor | None = None,
    rsqrt_1: LogicalTensor | None = None,
    mul_3: LogicalTensor | None = None,
    to_3: LogicalTensor | None = None,
    mul_4: LogicalTensor | None = None,
    to_4: LogicalTensor | None = None,
    to_5: LogicalTensor | None = None,
    to_6: LogicalTensor | None = None,
    reshape_1: LogicalTensor | None = None,
    to_7: LogicalTensor | None = None,
    reshape_2: LogicalTensor | None = None,
    select: LogicalTensor | None = None,
    select_1: LogicalTensor | None = None,
    mul_5: LogicalTensor | None = None,
    select_2: LogicalTensor | None = None,
    select_3: LogicalTensor | None = None,
    mul_6: LogicalTensor | None = None,
    add_4: LogicalTensor | None = None,
    select_4: LogicalTensor | None = None,
    select_5: LogicalTensor | None = None,
    mul_7: LogicalTensor | None = None,
    select_6: LogicalTensor | None = None,
    select_7: LogicalTensor | None = None,
    mul_8: LogicalTensor | None = None,
    add_5: LogicalTensor | None = None,
    reshape_3: LogicalTensor | None = None,
    type_as: LogicalTensor | None = None,
    reshape_4: LogicalTensor | None = None,
    type_as_1: LogicalTensor | None = None,
    scaled_dot_product_attention: LogicalTensor | None = None,
    permute_1: LogicalTensor | None = None,
    reshape_5: LogicalTensor | None = None,
    getitem_5: LogicalTensor | None = None,
    getitem_6: LogicalTensor | None = None,
    silu: LogicalTensor | None = None,
    mul_9: LogicalTensor | None = None,
    cat: LogicalTensor | None = None,
    linear_1: LogicalTensor | None = None,
    mul_10: LogicalTensor | None = None,
    add_6: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> FluxSingleBlockTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('add_6',)))
    tensors = FluxSingleBlockTensors(
        p_linear1_weight=_bind_tensor(
            p_linear1_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"single_blocks.{layer_idx}.linear1.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"single_blocks.{layer_idx}.linear1.weight", dtype='float32', shape=(36864, 4096)),
                layout=_quantized_weight_layout(f"single_blocks.{layer_idx}.linear1.weight", dtype='float32', shape=(36864, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_linear1_weight' in request_state_outputs,
            ),
        ),
        p_linear2_weight=_bind_tensor(
            p_linear2_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"single_blocks.{layer_idx}.linear2.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"single_blocks.{layer_idx}.linear2.weight", dtype='float32', shape=(4096, 16384)),
                layout=_quantized_weight_layout(f"single_blocks.{layer_idx}.linear2.weight", dtype='float32', shape=(4096, 16384)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_linear2_weight' in request_state_outputs,
            ),
        ),
        p_norm_query_norm_scale=_bind_tensor(
            p_norm_query_norm_scale,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"single_blocks.{layer_idx}.norm.query_norm.scale",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"single_blocks.{layer_idx}.norm.query_norm.scale", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout(f"single_blocks.{layer_idx}.norm.query_norm.scale", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_norm_query_norm_scale' in request_state_outputs,
            ),
        ),
        p_norm_key_norm_scale=_bind_tensor(
            p_norm_key_norm_scale,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key=f"single_blocks.{layer_idx}.norm.key_norm.scale",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"single_blocks.{layer_idx}.norm.key_norm.scale", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout(f"single_blocks.{layer_idx}.norm.key_norm.scale", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_norm_key_norm_scale' in request_state_outputs,
            ),
        ),
        hidden_states=_bind_tensor(
            hidden_states,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='hidden_states' in request_state_outputs,
            ),
        ),
        pe=_bind_tensor(
            pe,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len + image_seq_len, 64, 2, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='pe' in request_state_outputs,
            ),
        ),
        mod_shift=_bind_tensor(
            mod_shift,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='mod_shift' in request_state_outputs,
            ),
        ),
        mod_scale=_bind_tensor(
            mod_scale,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='mod_scale' in request_state_outputs,
            ),
        ),
        mod_gate=_bind_tensor(
            mod_gate,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='mod_gate' in request_state_outputs,
            ),
        ),
        add=_bind_tensor(
            add,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add' in request_state_outputs,
            ),
        ),
        layer_norm=_bind_tensor(
            layer_norm,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='layer_norm',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_norm' in request_state_outputs,
            ),
        ),
        mul=_bind_tensor(
            mul,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul' in request_state_outputs,
            ),
        ),
        add_1=_bind_tensor(
            add_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_1' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            linear,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 36864)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear' in request_state_outputs,
            ),
        ),
        getitem=_bind_tensor(
            getitem,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 12288)),
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
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 24576)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_1' in request_state_outputs,
            ),
        ),
        reshape=_bind_tensor(
            reshape,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 3, 32, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape' in request_state_outputs,
            ),
        ),
        permute=_bind_tensor(
            permute,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(3, 1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute' in request_state_outputs,
            ),
        ),
        getitem_2=_bind_tensor(
            getitem_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
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
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
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
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_4' in request_state_outputs,
            ),
        ),
        to=_bind_tensor(
            to,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to' in request_state_outputs,
            ),
        ),
        pow_1=_bind_tensor(
            pow_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='pow_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_1' in request_state_outputs,
            ),
        ),
        mean=_bind_tensor(
            mean,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mean',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mean' in request_state_outputs,
            ),
        ),
        add_2=_bind_tensor(
            add_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_2' in request_state_outputs,
            ),
        ),
        rsqrt=_bind_tensor(
            rsqrt,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rsqrt',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rsqrt' in request_state_outputs,
            ),
        ),
        mul_1=_bind_tensor(
            mul_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_1' in request_state_outputs,
            ),
        ),
        to_1=_bind_tensor(
            to_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_1' in request_state_outputs,
            ),
        ),
        mul_2=_bind_tensor(
            mul_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_2' in request_state_outputs,
            ),
        ),
        to_2=_bind_tensor(
            to_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_2' in request_state_outputs,
            ),
        ),
        pow_2=_bind_tensor(
            pow_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='pow_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_2' in request_state_outputs,
            ),
        ),
        mean_1=_bind_tensor(
            mean_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mean_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mean_1' in request_state_outputs,
            ),
        ),
        add_3=_bind_tensor(
            add_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_3' in request_state_outputs,
            ),
        ),
        rsqrt_1=_bind_tensor(
            rsqrt_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rsqrt_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rsqrt_1' in request_state_outputs,
            ),
        ),
        mul_3=_bind_tensor(
            mul_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_3' in request_state_outputs,
            ),
        ),
        to_3=_bind_tensor(
            to_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_3' in request_state_outputs,
            ),
        ),
        mul_4=_bind_tensor(
            mul_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_4' in request_state_outputs,
            ),
        ),
        to_4=_bind_tensor(
            to_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_4' in request_state_outputs,
            ),
        ),
        to_5=_bind_tensor(
            to_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_5' in request_state_outputs,
            ),
        ),
        to_6=_bind_tensor(
            to_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_6' in request_state_outputs,
            ),
        ),
        reshape_1=_bind_tensor(
            reshape_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_1' in request_state_outputs,
            ),
        ),
        to_7=_bind_tensor(
            to_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_7' in request_state_outputs,
            ),
        ),
        reshape_2=_bind_tensor(
            reshape_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_2' in request_state_outputs,
            ),
        ),
        select=_bind_tensor(
            select,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select' in request_state_outputs,
            ),
        ),
        select_1=_bind_tensor(
            select_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_1' in request_state_outputs,
            ),
        ),
        mul_5=_bind_tensor(
            mul_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_5' in request_state_outputs,
            ),
        ),
        select_2=_bind_tensor(
            select_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_2',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_2' in request_state_outputs,
            ),
        ),
        select_3=_bind_tensor(
            select_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_3' in request_state_outputs,
            ),
        ),
        mul_6=_bind_tensor(
            mul_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_6' in request_state_outputs,
            ),
        ),
        add_4=_bind_tensor(
            add_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_4' in request_state_outputs,
            ),
        ),
        select_4=_bind_tensor(
            select_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_4' in request_state_outputs,
            ),
        ),
        select_5=_bind_tensor(
            select_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_5' in request_state_outputs,
            ),
        ),
        mul_7=_bind_tensor(
            mul_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_7' in request_state_outputs,
            ),
        ),
        select_6=_bind_tensor(
            select_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 1, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_6' in request_state_outputs,
            ),
        ),
        select_7=_bind_tensor(
            select_7,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='select_7',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='select_7' in request_state_outputs,
            ),
        ),
        mul_8=_bind_tensor(
            mul_8,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_8',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_8' in request_state_outputs,
            ),
        ),
        add_5=_bind_tensor(
            add_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 64, 2)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_5' in request_state_outputs,
            ),
        ),
        reshape_3=_bind_tensor(
            reshape_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_3',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_3' in request_state_outputs,
            ),
        ),
        type_as=_bind_tensor(
            type_as,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='type_as',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='type_as' in request_state_outputs,
            ),
        ),
        reshape_4=_bind_tensor(
            reshape_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_4',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_4' in request_state_outputs,
            ),
        ),
        type_as_1=_bind_tensor(
            type_as_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='type_as_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='type_as_1' in request_state_outputs,
            ),
        ),
        scaled_dot_product_attention=_bind_tensor(
            scaled_dot_product_attention,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='scaled_dot_product_attention',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, 32, text_seq_len + image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='scaled_dot_product_attention' in request_state_outputs,
            ),
        ),
        permute_1=_bind_tensor(
            permute_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 32, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute_1' in request_state_outputs,
            ),
        ),
        reshape_5=_bind_tensor(
            reshape_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_5' in request_state_outputs,
            ),
        ),
        getitem_5=_bind_tensor(
            getitem_5,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_5',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_5' in request_state_outputs,
            ),
        ),
        getitem_6=_bind_tensor(
            getitem_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_6' in request_state_outputs,
            ),
        ),
        silu=_bind_tensor(
            silu,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='silu',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='silu' in request_state_outputs,
            ),
        ),
        mul_9=_bind_tensor(
            mul_9,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_9',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 12288)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_9' in request_state_outputs,
            ),
        ),
        cat=_bind_tensor(
            cat,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 16384)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat' in request_state_outputs,
            ),
        ),
        linear_1=_bind_tensor(
            linear_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_1',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_1' in request_state_outputs,
            ),
        ),
        mul_10=_bind_tensor(
            mul_10,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_10',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_10' in request_state_outputs,
            ),
        ),
        add_6=_bind_tensor(
            add_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_6',
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_6' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    _bind_alias_source(tensors.getitem, tensors.reshape)
    _bind_alias_source(tensors.getitem_2, tensors.to)
    _bind_alias_source(tensors.mul_1, tensors.to_1)
    _bind_alias_source(tensors.getitem_3, tensors.to_2)
    _bind_alias_source(tensors.mul_3, tensors.to_3)
    _bind_alias_source(tensors.mul_2, tensors.to_4)
    _bind_alias_source(tensors.mul_4, tensors.to_5)
    _bind_alias_source(tensors.to_4, tensors.to_6)
    _bind_alias_source(tensors.to_6, tensors.reshape_1)
    _bind_alias_source(tensors.to_5, tensors.to_7)
    _bind_alias_source(tensors.to_7, tensors.reshape_2)
    _bind_alias_source(tensors.add_4, tensors.reshape_3)
    _bind_alias_source(tensors.reshape_3, tensors.type_as)
    _bind_alias_source(tensors.add_5, tensors.reshape_4)
    _bind_alias_source(tensors.reshape_4, tensors.type_as_1)
    _bind_alias_source(tensors.permute_1, tensors.reshape_5)
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
