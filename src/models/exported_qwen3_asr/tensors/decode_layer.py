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
)


@dataclass(frozen=True, slots=True)
class DecodeLayerTensors:
    p_attn_q_proj_weight: LogicalTensor
    p_attn_k_proj_weight: LogicalTensor
    p_attn_v_proj_weight: LogicalTensor
    p_attn_o_proj_weight: LogicalTensor
    p_attn_q_norm_weight: LogicalTensor
    p_attn_k_norm_weight: LogicalTensor
    p_mlp_gate_proj_weight: LogicalTensor
    p_mlp_up_proj_weight: LogicalTensor
    p_mlp_down_proj_weight: LogicalTensor
    p_input_layernorm_weight: LogicalTensor
    p_post_attention_layernorm_weight: LogicalTensor
    hidden_states: LogicalTensor
    position_embeddings_0: LogicalTensor
    position_embeddings_1: LogicalTensor
    cache_position: LogicalTensor
    to: LogicalTensor
    pow_1: LogicalTensor
    mean: LogicalTensor
    add: LogicalTensor
    rsqrt: LogicalTensor
    mul: LogicalTensor
    to_1: LogicalTensor
    mul_1: LogicalTensor
    linear: LogicalTensor
    view: LogicalTensor
    to_2: LogicalTensor
    pow_2: LogicalTensor
    mean_1: LogicalTensor
    add_1: LogicalTensor
    rsqrt_1: LogicalTensor
    mul_2: LogicalTensor
    to_3: LogicalTensor
    mul_3: LogicalTensor
    transpose: LogicalTensor
    linear_1: LogicalTensor
    view_1: LogicalTensor
    to_4: LogicalTensor
    pow_3: LogicalTensor
    mean_2: LogicalTensor
    add_2: LogicalTensor
    rsqrt_2: LogicalTensor
    mul_4: LogicalTensor
    to_5: LogicalTensor
    mul_5: LogicalTensor
    transpose_1: LogicalTensor
    linear_2: LogicalTensor
    view_2: LogicalTensor
    transpose_2: LogicalTensor
    unsqueeze: LogicalTensor
    unsqueeze_1: LogicalTensor
    mul_6: LogicalTensor
    slice_1: LogicalTensor
    slice_2: LogicalTensor
    neg: LogicalTensor
    cat: LogicalTensor
    mul_7: LogicalTensor
    add_3: LogicalTensor
    mul_8: LogicalTensor
    slice_3: LogicalTensor
    slice_4: LogicalTensor
    neg_1: LogicalTensor
    cat_1: LogicalTensor
    mul_9: LogicalTensor
    add_4: LogicalTensor
    index_copy: LogicalTensor
    index_copy_1: LogicalTensor
    scaled_dot_product_attention: LogicalTensor
    transpose_3: LogicalTensor
    reshape: LogicalTensor
    linear_3: LogicalTensor
    add_5: LogicalTensor
    to_6: LogicalTensor
    pow_4: LogicalTensor
    mean_3: LogicalTensor
    add_6: LogicalTensor
    rsqrt_3: LogicalTensor
    mul_10: LogicalTensor
    to_7: LogicalTensor
    mul_11: LogicalTensor
    linear_4: LogicalTensor
    silu: LogicalTensor
    linear_5: LogicalTensor
    mul_12: LogicalTensor
    linear_6: LogicalTensor
    add_7: LogicalTensor


DECODE_LAYER_OUTPUT: str = 'add_7'


def create_decode_layer(
    prefix: str,
    layer_idx: int,
    *,
    p_attn_q_proj_weight: LogicalTensor | None = None,
    p_attn_k_proj_weight: LogicalTensor | None = None,
    p_attn_v_proj_weight: LogicalTensor | None = None,
    p_attn_o_proj_weight: LogicalTensor | None = None,
    p_attn_q_norm_weight: LogicalTensor | None = None,
    p_attn_k_norm_weight: LogicalTensor | None = None,
    p_mlp_gate_proj_weight: LogicalTensor | None = None,
    p_mlp_up_proj_weight: LogicalTensor | None = None,
    p_mlp_down_proj_weight: LogicalTensor | None = None,
    p_input_layernorm_weight: LogicalTensor | None = None,
    p_post_attention_layernorm_weight: LogicalTensor | None = None,
    hidden_states: LogicalTensor | None = None,
    position_embeddings_0: LogicalTensor | None = None,
    position_embeddings_1: LogicalTensor | None = None,
    cache_position: LogicalTensor | None = None,
    to: LogicalTensor | None = None,
    pow_1: LogicalTensor | None = None,
    mean: LogicalTensor | None = None,
    add: LogicalTensor | None = None,
    rsqrt: LogicalTensor | None = None,
    mul: LogicalTensor | None = None,
    to_1: LogicalTensor | None = None,
    mul_1: LogicalTensor | None = None,
    linear: LogicalTensor | None = None,
    view: LogicalTensor | None = None,
    to_2: LogicalTensor | None = None,
    pow_2: LogicalTensor | None = None,
    mean_1: LogicalTensor | None = None,
    add_1: LogicalTensor | None = None,
    rsqrt_1: LogicalTensor | None = None,
    mul_2: LogicalTensor | None = None,
    to_3: LogicalTensor | None = None,
    mul_3: LogicalTensor | None = None,
    transpose: LogicalTensor | None = None,
    linear_1: LogicalTensor | None = None,
    view_1: LogicalTensor | None = None,
    to_4: LogicalTensor | None = None,
    pow_3: LogicalTensor | None = None,
    mean_2: LogicalTensor | None = None,
    add_2: LogicalTensor | None = None,
    rsqrt_2: LogicalTensor | None = None,
    mul_4: LogicalTensor | None = None,
    to_5: LogicalTensor | None = None,
    mul_5: LogicalTensor | None = None,
    transpose_1: LogicalTensor | None = None,
    linear_2: LogicalTensor | None = None,
    view_2: LogicalTensor | None = None,
    transpose_2: LogicalTensor | None = None,
    unsqueeze: LogicalTensor | None = None,
    unsqueeze_1: LogicalTensor | None = None,
    mul_6: LogicalTensor | None = None,
    slice_1: LogicalTensor | None = None,
    slice_2: LogicalTensor | None = None,
    neg: LogicalTensor | None = None,
    cat: LogicalTensor | None = None,
    mul_7: LogicalTensor | None = None,
    add_3: LogicalTensor | None = None,
    mul_8: LogicalTensor | None = None,
    slice_3: LogicalTensor | None = None,
    slice_4: LogicalTensor | None = None,
    neg_1: LogicalTensor | None = None,
    cat_1: LogicalTensor | None = None,
    mul_9: LogicalTensor | None = None,
    add_4: LogicalTensor | None = None,
    index_copy: LogicalTensor | None = None,
    index_copy_1: LogicalTensor | None = None,
    scaled_dot_product_attention: LogicalTensor | None = None,
    transpose_3: LogicalTensor | None = None,
    reshape: LogicalTensor | None = None,
    linear_3: LogicalTensor | None = None,
    add_5: LogicalTensor | None = None,
    to_6: LogicalTensor | None = None,
    pow_4: LogicalTensor | None = None,
    mean_3: LogicalTensor | None = None,
    add_6: LogicalTensor | None = None,
    rsqrt_3: LogicalTensor | None = None,
    mul_10: LogicalTensor | None = None,
    to_7: LogicalTensor | None = None,
    mul_11: LogicalTensor | None = None,
    linear_4: LogicalTensor | None = None,
    silu: LogicalTensor | None = None,
    linear_5: LogicalTensor | None = None,
    mul_12: LogicalTensor | None = None,
    linear_6: LogicalTensor | None = None,
    add_7: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> DecodeLayerTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('add_7',)))
    tensors = DecodeLayerTensors(
        p_attn_q_proj_weight=_bind_tensor(
            p_attn_q_proj_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.q_proj.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(2048, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_q_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_k_proj_weight=_bind_tensor(
            p_attn_k_proj_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.k_proj.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(1024, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_k_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_v_proj_weight=_bind_tensor(
            p_attn_v_proj_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.v_proj.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(1024, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_v_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_o_proj_weight=_bind_tensor(
            p_attn_o_proj_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.o_proj.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(1024, 2048)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_o_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_q_norm_weight=_bind_tensor(
            p_attn_q_norm_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.q_norm.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(128,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_q_norm_weight' in request_state_outputs,
            ),
        ),
        p_attn_k_norm_weight=_bind_tensor(
            p_attn_k_norm_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.k_norm.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(128,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_k_norm_weight' in request_state_outputs,
            ),
        ),
        p_mlp_gate_proj_weight=_bind_tensor(
            p_mlp_gate_proj_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.model.layers.{layer_idx}.mlp.gate_proj.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(3072, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_mlp_gate_proj_weight' in request_state_outputs,
            ),
        ),
        p_mlp_up_proj_weight=_bind_tensor(
            p_mlp_up_proj_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.model.layers.{layer_idx}.mlp.up_proj.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(3072, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_mlp_up_proj_weight' in request_state_outputs,
            ),
        ),
        p_mlp_down_proj_weight=_bind_tensor(
            p_mlp_down_proj_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.model.layers.{layer_idx}.mlp.down_proj.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(1024, 3072)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_mlp_down_proj_weight' in request_state_outputs,
            ),
        ),
        p_input_layernorm_weight=_bind_tensor(
            p_input_layernorm_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.model.layers.{layer_idx}.input_layernorm.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(1024,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_input_layernorm_weight' in request_state_outputs,
            ),
        ),
        p_post_attention_layernorm_weight=_bind_tensor(
            p_post_attention_layernorm_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.model.layers.{layer_idx}.post_attention_layernorm.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(1024,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_post_attention_layernorm_weight' in request_state_outputs,
            ),
        ),
        hidden_states=_bind_tensor(
            hidden_states,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='hidden_states' in request_state_outputs,
            ),
        ),
        position_embeddings_0=_bind_tensor(
            position_embeddings_0,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='position_embeddings_0' in request_state_outputs,
            ),
        ),
        position_embeddings_1=_bind_tensor(
            position_embeddings_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='position_embeddings_1' in request_state_outputs,
            ),
        ),
        cache_position=_bind_tensor(
            cache_position,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='int64', shape=(1,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='cache_position' in request_state_outputs,
            ),
        ),
        to=_bind_tensor(
            to,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='to',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
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
                checkpoint_key=None,
                reference_key='pow_1',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
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
                checkpoint_key=None,
                reference_key='mean',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mean' in request_state_outputs,
            ),
        ),
        add=_bind_tensor(
            add,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add' in request_state_outputs,
            ),
        ),
        rsqrt=_bind_tensor(
            rsqrt,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='rsqrt',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rsqrt' in request_state_outputs,
            ),
        ),
        mul=_bind_tensor(
            mul,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul' in request_state_outputs,
            ),
        ),
        to_1=_bind_tensor(
            to_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='to_1',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_1' in request_state_outputs,
            ),
        ),
        mul_1=_bind_tensor(
            mul_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_1',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_1' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            linear,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 2048)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear' in request_state_outputs,
            ),
        ),
        view=_bind_tensor(
            view,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='view',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 16, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='view' in request_state_outputs,
            ),
        ),
        to_2=_bind_tensor(
            to_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='to_2',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 16, 128)),
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
                checkpoint_key=None,
                reference_key='pow_2',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 16, 128)),
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
                checkpoint_key=None,
                reference_key='mean_1',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 16, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mean_1' in request_state_outputs,
            ),
        ),
        add_1=_bind_tensor(
            add_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add_1',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 16, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_1' in request_state_outputs,
            ),
        ),
        rsqrt_1=_bind_tensor(
            rsqrt_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='rsqrt_1',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 16, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rsqrt_1' in request_state_outputs,
            ),
        ),
        mul_2=_bind_tensor(
            mul_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_2',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 16, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_2' in request_state_outputs,
            ),
        ),
        to_3=_bind_tensor(
            to_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='to_3',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 16, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_3' in request_state_outputs,
            ),
        ),
        mul_3=_bind_tensor(
            mul_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_3',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 16, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_3' in request_state_outputs,
            ),
        ),
        transpose=_bind_tensor(
            transpose,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='transpose',
                spec=TensorSpec(dtype='float32', shape=(1, 16, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='transpose' in request_state_outputs,
            ),
        ),
        linear_1=_bind_tensor(
            linear_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_1',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_1' in request_state_outputs,
            ),
        ),
        view_1=_bind_tensor(
            view_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='view_1',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 8, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='view_1' in request_state_outputs,
            ),
        ),
        to_4=_bind_tensor(
            to_4,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='to_4',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 8, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_4' in request_state_outputs,
            ),
        ),
        pow_3=_bind_tensor(
            pow_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='pow_3',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 8, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_3' in request_state_outputs,
            ),
        ),
        mean_2=_bind_tensor(
            mean_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mean_2',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 8, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mean_2' in request_state_outputs,
            ),
        ),
        add_2=_bind_tensor(
            add_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add_2',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 8, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_2' in request_state_outputs,
            ),
        ),
        rsqrt_2=_bind_tensor(
            rsqrt_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='rsqrt_2',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 8, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rsqrt_2' in request_state_outputs,
            ),
        ),
        mul_4=_bind_tensor(
            mul_4,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_4',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 8, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_4' in request_state_outputs,
            ),
        ),
        to_5=_bind_tensor(
            to_5,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='to_5',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 8, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_5' in request_state_outputs,
            ),
        ),
        mul_5=_bind_tensor(
            mul_5,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_5',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 8, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_5' in request_state_outputs,
            ),
        ),
        transpose_1=_bind_tensor(
            transpose_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='transpose_1',
                spec=TensorSpec(dtype='float32', shape=(1, 8, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='transpose_1' in request_state_outputs,
            ),
        ),
        linear_2=_bind_tensor(
            linear_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_2',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_2' in request_state_outputs,
            ),
        ),
        view_2=_bind_tensor(
            view_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='view_2',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 8, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='view_2' in request_state_outputs,
            ),
        ),
        transpose_2=_bind_tensor(
            transpose_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='transpose_2',
                spec=TensorSpec(dtype='float32', shape=(1, 8, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='transpose_2' in request_state_outputs,
            ),
        ),
        unsqueeze=_bind_tensor(
            unsqueeze,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='unsqueeze',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze' in request_state_outputs,
            ),
        ),
        unsqueeze_1=_bind_tensor(
            unsqueeze_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='unsqueeze_1',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze_1' in request_state_outputs,
            ),
        ),
        mul_6=_bind_tensor(
            mul_6,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_6',
                spec=TensorSpec(dtype='float32', shape=(1, 16, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_6' in request_state_outputs,
            ),
        ),
        slice_1=_bind_tensor(
            slice_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='slice_1',
                spec=TensorSpec(dtype='float32', shape=(1, 16, 1, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='slice_1' in request_state_outputs,
            ),
        ),
        slice_2=_bind_tensor(
            slice_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='slice_2',
                spec=TensorSpec(dtype='float32', shape=(1, 16, 1, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='slice_2' in request_state_outputs,
            ),
        ),
        neg=_bind_tensor(
            neg,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='neg',
                spec=TensorSpec(dtype='float32', shape=(1, 16, 1, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='neg' in request_state_outputs,
            ),
        ),
        cat=_bind_tensor(
            cat,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='cat',
                spec=TensorSpec(dtype='float32', shape=(1, 16, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat' in request_state_outputs,
            ),
        ),
        mul_7=_bind_tensor(
            mul_7,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_7',
                spec=TensorSpec(dtype='float32', shape=(1, 16, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_7' in request_state_outputs,
            ),
        ),
        add_3=_bind_tensor(
            add_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add_3',
                spec=TensorSpec(dtype='float32', shape=(1, 16, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_3' in request_state_outputs,
            ),
        ),
        mul_8=_bind_tensor(
            mul_8,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_8',
                spec=TensorSpec(dtype='float32', shape=(1, 8, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_8' in request_state_outputs,
            ),
        ),
        slice_3=_bind_tensor(
            slice_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='slice_3',
                spec=TensorSpec(dtype='float32', shape=(1, 8, 1, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='slice_3' in request_state_outputs,
            ),
        ),
        slice_4=_bind_tensor(
            slice_4,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='slice_4',
                spec=TensorSpec(dtype='float32', shape=(1, 8, 1, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='slice_4' in request_state_outputs,
            ),
        ),
        neg_1=_bind_tensor(
            neg_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='neg_1',
                spec=TensorSpec(dtype='float32', shape=(1, 8, 1, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='neg_1' in request_state_outputs,
            ),
        ),
        cat_1=_bind_tensor(
            cat_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='cat_1',
                spec=TensorSpec(dtype='float32', shape=(1, 8, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat_1' in request_state_outputs,
            ),
        ),
        mul_9=_bind_tensor(
            mul_9,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_9',
                spec=TensorSpec(dtype='float32', shape=(1, 8, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_9' in request_state_outputs,
            ),
        ),
        add_4=_bind_tensor(
            add_4,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add_4',
                spec=TensorSpec(dtype='float32', shape=(1, 8, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_4' in request_state_outputs,
            ),
        ),
        index_copy=_bind_tensor(
            index_copy,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='index_copy',
                spec=TensorSpec(dtype='float32', shape=(1, 8, 215, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='index_copy' in request_state_outputs,
            ),
        ),
        index_copy_1=_bind_tensor(
            index_copy_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='index_copy_1',
                spec=TensorSpec(dtype='float32', shape=(1, 8, 215, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='index_copy_1' in request_state_outputs,
            ),
        ),
        scaled_dot_product_attention=_bind_tensor(
            scaled_dot_product_attention,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='scaled_dot_product_attention',
                spec=TensorSpec(dtype='float32', shape=(1, 16, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='scaled_dot_product_attention' in request_state_outputs,
            ),
        ),
        transpose_3=_bind_tensor(
            transpose_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='transpose_3',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 16, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='transpose_3' in request_state_outputs,
            ),
        ),
        reshape=_bind_tensor(
            reshape,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='reshape',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 2048)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape' in request_state_outputs,
            ),
        ),
        linear_3=_bind_tensor(
            linear_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_3',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_3' in request_state_outputs,
            ),
        ),
        add_5=_bind_tensor(
            add_5,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add_5',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_5' in request_state_outputs,
            ),
        ),
        to_6=_bind_tensor(
            to_6,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='to_6',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_6' in request_state_outputs,
            ),
        ),
        pow_4=_bind_tensor(
            pow_4,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='pow_4',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_4' in request_state_outputs,
            ),
        ),
        mean_3=_bind_tensor(
            mean_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mean_3',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mean_3' in request_state_outputs,
            ),
        ),
        add_6=_bind_tensor(
            add_6,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add_6',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_6' in request_state_outputs,
            ),
        ),
        rsqrt_3=_bind_tensor(
            rsqrt_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='rsqrt_3',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rsqrt_3' in request_state_outputs,
            ),
        ),
        mul_10=_bind_tensor(
            mul_10,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_10',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_10' in request_state_outputs,
            ),
        ),
        to_7=_bind_tensor(
            to_7,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='to_7',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_7' in request_state_outputs,
            ),
        ),
        mul_11=_bind_tensor(
            mul_11,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_11',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_11' in request_state_outputs,
            ),
        ),
        linear_4=_bind_tensor(
            linear_4,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_4',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 3072)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_4' in request_state_outputs,
            ),
        ),
        silu=_bind_tensor(
            silu,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='silu',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 3072)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='silu' in request_state_outputs,
            ),
        ),
        linear_5=_bind_tensor(
            linear_5,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_5',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 3072)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_5' in request_state_outputs,
            ),
        ),
        mul_12=_bind_tensor(
            mul_12,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_12',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 3072)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_12' in request_state_outputs,
            ),
        ),
        linear_6=_bind_tensor(
            linear_6,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_6',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_6' in request_state_outputs,
            ),
        ),
        add_7=_bind_tensor(
            add_7,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add_7',
                spec=TensorSpec(dtype='float32', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_7' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    _bind_alias_source(tensors.hidden_states, tensors.to)
    _bind_alias_source(tensors.mul, tensors.to_1)
    _bind_alias_source(tensors.linear, tensors.view)
    _bind_alias_source(tensors.view, tensors.to_2)
    _bind_alias_source(tensors.mul_2, tensors.to_3)
    _bind_alias_source(tensors.linear_1, tensors.view_1)
    _bind_alias_source(tensors.view_1, tensors.to_4)
    _bind_alias_source(tensors.mul_4, tensors.to_5)
    _bind_alias_source(tensors.linear_2, tensors.view_2)
    _bind_alias_source(tensors.position_embeddings_0, tensors.unsqueeze)
    _bind_alias_source(tensors.position_embeddings_1, tensors.unsqueeze_1)
    _bind_alias_source(tensors.transpose_3, tensors.reshape)
    _bind_alias_source(tensors.add_5, tensors.to_6)
    _bind_alias_source(tensors.mul_10, tensors.to_7)
    return tensors


def _declare_tensor(
    *,
    spec: TensorSpec,
    role: TensorRole,
    memory: MemoryClass,
    lifetime: TensorLifetime,
    layout: TensorLayout = CONTIGUOUS_LAYOUT,
    checkpoint_key: str | None = None,
    reference_key: str | None = None,
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
        checkpoint_key=checkpoint_key,
        reference_key=reference_key,
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
