"""Generated tensor declarations."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from torch2vk.runtime.logical import (
    ComparePolicy,
    LogicalTensor,
    MemoryClass,
    PyTorchProbe,
    TensorLifetime,
    TensorRole,
)
from torch2vk.vulkan.types import TensorSpec


@dataclass(frozen=True, slots=True)
class TextLayerTensors:
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
    contiguous: LogicalTensor
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


TEXT_LAYER_OUTPUT: str = 'add_7'


def create_text_layer(
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
    contiguous: LogicalTensor | None = None,
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
) -> TextLayerTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('add_7',)))
    return TextLayerTensors(
        p_attn_q_proj_weight=_bind_tensor(
            p_attn_q_proj_weight,
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.self_attn.q_proj.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(2048, 1024)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_q_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_k_proj_weight=_bind_tensor(
            p_attn_k_proj_weight,
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.self_attn.k_proj.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(1024, 1024)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_k_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_v_proj_weight=_bind_tensor(
            p_attn_v_proj_weight,
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.self_attn.v_proj.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(1024, 1024)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_v_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_o_proj_weight=_bind_tensor(
            p_attn_o_proj_weight,
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.self_attn.o_proj.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(1024, 2048)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_o_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_q_norm_weight=_bind_tensor(
            p_attn_q_norm_weight,
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.self_attn.q_norm.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(128,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_q_norm_weight' in request_state_outputs,
            ),
        ),
        p_attn_k_norm_weight=_bind_tensor(
            p_attn_k_norm_weight,
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.self_attn.k_norm.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(128,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_k_norm_weight' in request_state_outputs,
            ),
        ),
        p_mlp_gate_proj_weight=_bind_tensor(
            p_mlp_gate_proj_weight,
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.mlp.gate_proj.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(3072, 1024)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_mlp_gate_proj_weight' in request_state_outputs,
            ),
        ),
        p_mlp_up_proj_weight=_bind_tensor(
            p_mlp_up_proj_weight,
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.mlp.up_proj.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(3072, 1024)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_mlp_up_proj_weight' in request_state_outputs,
            ),
        ),
        p_mlp_down_proj_weight=_bind_tensor(
            p_mlp_down_proj_weight,
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.mlp.down_proj.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(1024, 3072)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_mlp_down_proj_weight' in request_state_outputs,
            ),
        ),
        p_input_layernorm_weight=_bind_tensor(
            p_input_layernorm_weight,
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.input_layernorm.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(1024,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_input_layernorm_weight' in request_state_outputs,
            ),
        ),
        p_post_attention_layernorm_weight=_bind_tensor(
            p_post_attention_layernorm_weight,
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.post_attention_layernorm.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(1024,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_post_attention_layernorm_weight' in request_state_outputs,
            ),
        ),
        hidden_states=_bind_tensor(
            hidden_states,
            _declare_tensor(
            name=f"{prefix}.hidden_states",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='hidden_states' in request_state_outputs,
            ),
        ),
        position_embeddings_0=_bind_tensor(
            position_embeddings_0,
            _declare_tensor(
            name=f"{prefix}.position_embeddings_0",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 128)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='position_embeddings_0' in request_state_outputs,
            ),
        ),
        position_embeddings_1=_bind_tensor(
            position_embeddings_1,
            _declare_tensor(
            name=f"{prefix}.position_embeddings_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 128)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='position_embeddings_1' in request_state_outputs,
            ),
        ),
        cache_position=_bind_tensor(
            cache_position,
            _declare_tensor(
            name=f"{prefix}.cache_position",
            spec=TensorSpec(dtype='int64', shape=(151,)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='cache_position' in request_state_outputs,
            ),
        ),
        to=_bind_tensor(
            to,
            _declare_tensor(
            name=f"{prefix}.to",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='to' in request_state_outputs,
            ),
        ),
        pow_1=_bind_tensor(
            pow_1,
            _declare_tensor(
            name=f"{prefix}.pow_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='pow_1' in request_state_outputs,
            ),
        ),
        mean=_bind_tensor(
            mean,
            _declare_tensor(
            name=f"{prefix}.mean",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mean' in request_state_outputs,
            ),
        ),
        add=_bind_tensor(
            add,
            _declare_tensor(
            name=f"{prefix}.add",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add' in request_state_outputs,
            ),
        ),
        rsqrt=_bind_tensor(
            rsqrt,
            _declare_tensor(
            name=f"{prefix}.rsqrt",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='rsqrt' in request_state_outputs,
            ),
        ),
        mul=_bind_tensor(
            mul,
            _declare_tensor(
            name=f"{prefix}.mul",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul' in request_state_outputs,
            ),
        ),
        to_1=_bind_tensor(
            to_1,
            _declare_tensor(
            name=f"{prefix}.to_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='to_1' in request_state_outputs,
            ),
        ),
        mul_1=_bind_tensor(
            mul_1,
            _declare_tensor(
            name=f"{prefix}.mul_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_1' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            linear,
            _declare_tensor(
            name=f"{prefix}.linear",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 2048)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear' in request_state_outputs,
            ),
        ),
        view=_bind_tensor(
            view,
            _declare_tensor(
            name=f"{prefix}.view",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='view' in request_state_outputs,
            ),
        ),
        to_2=_bind_tensor(
            to_2,
            _declare_tensor(
            name=f"{prefix}.to_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='to_2' in request_state_outputs,
            ),
        ),
        pow_2=_bind_tensor(
            pow_2,
            _declare_tensor(
            name=f"{prefix}.pow_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='pow_2' in request_state_outputs,
            ),
        ),
        mean_1=_bind_tensor(
            mean_1,
            _declare_tensor(
            name=f"{prefix}.mean_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mean_1' in request_state_outputs,
            ),
        ),
        add_1=_bind_tensor(
            add_1,
            _declare_tensor(
            name=f"{prefix}.add_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add_1' in request_state_outputs,
            ),
        ),
        rsqrt_1=_bind_tensor(
            rsqrt_1,
            _declare_tensor(
            name=f"{prefix}.rsqrt_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='rsqrt_1' in request_state_outputs,
            ),
        ),
        mul_2=_bind_tensor(
            mul_2,
            _declare_tensor(
            name=f"{prefix}.mul_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_2' in request_state_outputs,
            ),
        ),
        to_3=_bind_tensor(
            to_3,
            _declare_tensor(
            name=f"{prefix}.to_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='to_3' in request_state_outputs,
            ),
        ),
        mul_3=_bind_tensor(
            mul_3,
            _declare_tensor(
            name=f"{prefix}.mul_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_3' in request_state_outputs,
            ),
        ),
        transpose=_bind_tensor(
            transpose,
            _declare_tensor(
            name=f"{prefix}.transpose",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='transpose' in request_state_outputs,
            ),
        ),
        linear_1=_bind_tensor(
            linear_1,
            _declare_tensor(
            name=f"{prefix}.linear_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_1' in request_state_outputs,
            ),
        ),
        view_1=_bind_tensor(
            view_1,
            _declare_tensor(
            name=f"{prefix}.view_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='view_1' in request_state_outputs,
            ),
        ),
        to_4=_bind_tensor(
            to_4,
            _declare_tensor(
            name=f"{prefix}.to_4",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='to_4' in request_state_outputs,
            ),
        ),
        pow_3=_bind_tensor(
            pow_3,
            _declare_tensor(
            name=f"{prefix}.pow_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='pow_3' in request_state_outputs,
            ),
        ),
        mean_2=_bind_tensor(
            mean_2,
            _declare_tensor(
            name=f"{prefix}.mean_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mean_2' in request_state_outputs,
            ),
        ),
        add_2=_bind_tensor(
            add_2,
            _declare_tensor(
            name=f"{prefix}.add_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add_2' in request_state_outputs,
            ),
        ),
        rsqrt_2=_bind_tensor(
            rsqrt_2,
            _declare_tensor(
            name=f"{prefix}.rsqrt_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='rsqrt_2' in request_state_outputs,
            ),
        ),
        mul_4=_bind_tensor(
            mul_4,
            _declare_tensor(
            name=f"{prefix}.mul_4",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_4' in request_state_outputs,
            ),
        ),
        to_5=_bind_tensor(
            to_5,
            _declare_tensor(
            name=f"{prefix}.to_5",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='to_5' in request_state_outputs,
            ),
        ),
        mul_5=_bind_tensor(
            mul_5,
            _declare_tensor(
            name=f"{prefix}.mul_5",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_5' in request_state_outputs,
            ),
        ),
        transpose_1=_bind_tensor(
            transpose_1,
            _declare_tensor(
            name=f"{prefix}.transpose_1",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='transpose_1' in request_state_outputs,
            ),
        ),
        linear_2=_bind_tensor(
            linear_2,
            _declare_tensor(
            name=f"{prefix}.linear_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_2' in request_state_outputs,
            ),
        ),
        view_2=_bind_tensor(
            view_2,
            _declare_tensor(
            name=f"{prefix}.view_2",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 8, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='view_2' in request_state_outputs,
            ),
        ),
        transpose_2=_bind_tensor(
            transpose_2,
            _declare_tensor(
            name=f"{prefix}.transpose_2",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='transpose_2' in request_state_outputs,
            ),
        ),
        unsqueeze=_bind_tensor(
            unsqueeze,
            _declare_tensor(
            name=f"{prefix}.unsqueeze",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='unsqueeze' in request_state_outputs,
            ),
        ),
        unsqueeze_1=_bind_tensor(
            unsqueeze_1,
            _declare_tensor(
            name=f"{prefix}.unsqueeze_1",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='unsqueeze_1' in request_state_outputs,
            ),
        ),
        mul_6=_bind_tensor(
            mul_6,
            _declare_tensor(
            name=f"{prefix}.mul_6",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_6' in request_state_outputs,
            ),
        ),
        slice_1=_bind_tensor(
            slice_1,
            _declare_tensor(
            name=f"{prefix}.slice_1",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='slice_1' in request_state_outputs,
            ),
        ),
        slice_2=_bind_tensor(
            slice_2,
            _declare_tensor(
            name=f"{prefix}.slice_2",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='slice_2' in request_state_outputs,
            ),
        ),
        neg=_bind_tensor(
            neg,
            _declare_tensor(
            name=f"{prefix}.neg",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='neg' in request_state_outputs,
            ),
        ),
        cat=_bind_tensor(
            cat,
            _declare_tensor(
            name=f"{prefix}.cat",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='cat' in request_state_outputs,
            ),
        ),
        mul_7=_bind_tensor(
            mul_7,
            _declare_tensor(
            name=f"{prefix}.mul_7",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_7' in request_state_outputs,
            ),
        ),
        add_3=_bind_tensor(
            add_3,
            _declare_tensor(
            name=f"{prefix}.add_3",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add_3' in request_state_outputs,
            ),
        ),
        mul_8=_bind_tensor(
            mul_8,
            _declare_tensor(
            name=f"{prefix}.mul_8",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_8' in request_state_outputs,
            ),
        ),
        slice_3=_bind_tensor(
            slice_3,
            _declare_tensor(
            name=f"{prefix}.slice_3",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='slice_3' in request_state_outputs,
            ),
        ),
        slice_4=_bind_tensor(
            slice_4,
            _declare_tensor(
            name=f"{prefix}.slice_4",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='slice_4' in request_state_outputs,
            ),
        ),
        neg_1=_bind_tensor(
            neg_1,
            _declare_tensor(
            name=f"{prefix}.neg_1",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='neg_1' in request_state_outputs,
            ),
        ),
        cat_1=_bind_tensor(
            cat_1,
            _declare_tensor(
            name=f"{prefix}.cat_1",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='cat_1' in request_state_outputs,
            ),
        ),
        mul_9=_bind_tensor(
            mul_9,
            _declare_tensor(
            name=f"{prefix}.mul_9",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_9' in request_state_outputs,
            ),
        ),
        add_4=_bind_tensor(
            add_4,
            _declare_tensor(
            name=f"{prefix}.add_4",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add_4' in request_state_outputs,
            ),
        ),
        index_copy=_bind_tensor(
            index_copy,
            _declare_tensor(
            name=f"{prefix}.index_copy",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 215, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='index_copy' in request_state_outputs,
            ),
        ),
        index_copy_1=_bind_tensor(
            index_copy_1,
            _declare_tensor(
            name=f"{prefix}.index_copy_1",
            spec=TensorSpec(dtype='float32', shape=(1, 8, 215, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='index_copy_1' in request_state_outputs,
            ),
        ),
        scaled_dot_product_attention=_bind_tensor(
            scaled_dot_product_attention,
            _declare_tensor(
            name=f"{prefix}.scaled_dot_product_attention",
            spec=TensorSpec(dtype='float32', shape=(1, 16, 151, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='scaled_dot_product_attention' in request_state_outputs,
            ),
        ),
        transpose_3=_bind_tensor(
            transpose_3,
            _declare_tensor(
            name=f"{prefix}.transpose_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='transpose_3' in request_state_outputs,
            ),
        ),
        contiguous=_bind_tensor(
            contiguous,
            _declare_tensor(
            name=f"{prefix}.contiguous",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 16, 128)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='contiguous' in request_state_outputs,
            ),
        ),
        reshape=_bind_tensor(
            reshape,
            _declare_tensor(
            name=f"{prefix}.reshape",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 2048)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='reshape' in request_state_outputs,
            ),
        ),
        linear_3=_bind_tensor(
            linear_3,
            _declare_tensor(
            name=f"{prefix}.linear_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_3' in request_state_outputs,
            ),
        ),
        add_5=_bind_tensor(
            add_5,
            _declare_tensor(
            name=f"{prefix}.add_5",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add_5' in request_state_outputs,
            ),
        ),
        to_6=_bind_tensor(
            to_6,
            _declare_tensor(
            name=f"{prefix}.to_6",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='to_6' in request_state_outputs,
            ),
        ),
        pow_4=_bind_tensor(
            pow_4,
            _declare_tensor(
            name=f"{prefix}.pow_4",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='pow_4' in request_state_outputs,
            ),
        ),
        mean_3=_bind_tensor(
            mean_3,
            _declare_tensor(
            name=f"{prefix}.mean_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mean_3' in request_state_outputs,
            ),
        ),
        add_6=_bind_tensor(
            add_6,
            _declare_tensor(
            name=f"{prefix}.add_6",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add_6' in request_state_outputs,
            ),
        ),
        rsqrt_3=_bind_tensor(
            rsqrt_3,
            _declare_tensor(
            name=f"{prefix}.rsqrt_3",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='rsqrt_3' in request_state_outputs,
            ),
        ),
        mul_10=_bind_tensor(
            mul_10,
            _declare_tensor(
            name=f"{prefix}.mul_10",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_10' in request_state_outputs,
            ),
        ),
        to_7=_bind_tensor(
            to_7,
            _declare_tensor(
            name=f"{prefix}.to_7",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='to_7' in request_state_outputs,
            ),
        ),
        mul_11=_bind_tensor(
            mul_11,
            _declare_tensor(
            name=f"{prefix}.mul_11",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_11' in request_state_outputs,
            ),
        ),
        linear_4=_bind_tensor(
            linear_4,
            _declare_tensor(
            name=f"{prefix}.linear_4",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 3072)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_4' in request_state_outputs,
            ),
        ),
        silu=_bind_tensor(
            silu,
            _declare_tensor(
            name=f"{prefix}.silu",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 3072)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='silu' in request_state_outputs,
            ),
        ),
        linear_5=_bind_tensor(
            linear_5,
            _declare_tensor(
            name=f"{prefix}.linear_5",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 3072)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_5' in request_state_outputs,
            ),
        ),
        mul_12=_bind_tensor(
            mul_12,
            _declare_tensor(
            name=f"{prefix}.mul_12",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 3072)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='mul_12' in request_state_outputs,
            ),
        ),
        linear_6=_bind_tensor(
            linear_6,
            _declare_tensor(
            name=f"{prefix}.linear_6",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_6' in request_state_outputs,
            ),
        ),
        add_7=_bind_tensor(
            add_7,
            _declare_tensor(
            name=f"{prefix}.add_7",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add_7' in request_state_outputs,
            ),
        ),
    )


def _declare_tensor(
    *,
    name: str,
    spec: TensorSpec,
    role: TensorRole,
    memory: MemoryClass,
    lifetime: TensorLifetime,
    request_state: bool = False,
    compare: ComparePolicy | None = None,
    pytorch_probe: PyTorchProbe | None = None,
) -> LogicalTensor:
    if request_state:
        role = TensorRole.OUTPUT
        memory = MemoryClass.REQUEST_STATE
        lifetime = TensorLifetime.REQUEST
    return LogicalTensor(
        name=name,
        spec=spec,
        role=role,
        memory=memory,
        lifetime=lifetime,
        compare=compare,
        pytorch_probe=pytorch_probe,
    )


def _bind_tensor(
    bound: LogicalTensor | None,
    tensor: LogicalTensor,
) -> LogicalTensor:
    if bound is None:
        return tensor
    if bound.spec != tensor.spec:
        raise ValueError(f"{bound.name} spec {bound.spec} does not match {tensor.name} spec {tensor.spec}")
    return bound


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
