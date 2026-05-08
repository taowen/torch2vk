"""Generated tensor declarations."""

from __future__ import annotations

from collections.abc import Collection, Mapping
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
    p_input_layernorm_weight: LogicalTensor
    p_post_attention_layernorm_weight: LogicalTensor
    p_attn_q_proj_weight: LogicalTensor
    p_attn_k_proj_weight: LogicalTensor
    p_attn_v_proj_weight: LogicalTensor
    p_attn_o_proj_weight: LogicalTensor
    p_attn_q_norm_weight: LogicalTensor
    p_attn_k_norm_weight: LogicalTensor
    p_mlp_gate_proj_weight: LogicalTensor
    p_mlp_up_proj_weight: LogicalTensor
    p_mlp_down_proj_weight: LogicalTensor
    hidden_states: LogicalTensor
    cache_position: LogicalTensor
    position_embeddings_0: LogicalTensor
    position_embeddings_1: LogicalTensor
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


def create_text_layer(prefix: str, layer_idx: int, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> TextLayerTensors:
    _validate_bindings(bindings, frozenset(('p_input_layernorm_weight', 'p_post_attention_layernorm_weight', 'p_attn_q_proj_weight', 'p_attn_k_proj_weight', 'p_attn_v_proj_weight', 'p_attn_o_proj_weight', 'p_attn_q_norm_weight', 'p_attn_k_norm_weight', 'p_mlp_gate_proj_weight', 'p_mlp_up_proj_weight', 'p_mlp_down_proj_weight', 'hidden_states', 'cache_position', 'position_embeddings_0', 'position_embeddings_1', 'to', 'pow_1', 'mean', 'add', 'rsqrt', 'mul', 'to_1', 'mul_1', 'linear', 'view', 'to_2', 'pow_2', 'mean_1', 'add_1', 'rsqrt_1', 'mul_2', 'to_3', 'mul_3', 'transpose', 'linear_1', 'view_1', 'to_4', 'pow_3', 'mean_2', 'add_2', 'rsqrt_2', 'mul_4', 'to_5', 'mul_5', 'transpose_1', 'linear_2', 'view_2', 'transpose_2', 'unsqueeze', 'unsqueeze_1', 'mul_6', 'slice_1', 'slice_2', 'neg', 'cat', 'mul_7', 'add_3', 'mul_8', 'slice_3', 'slice_4', 'neg_1', 'cat_1', 'mul_9', 'add_4', 'index_copy', 'index_copy_1', 'scaled_dot_product_attention', 'transpose_3', 'contiguous', 'reshape', 'linear_3', 'add_5', 'to_6', 'pow_4', 'mean_3', 'add_6', 'rsqrt_3', 'mul_10', 'to_7', 'mul_11', 'linear_4', 'silu', 'linear_5', 'mul_12', 'linear_6', 'add_7')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('add_7',)))
    return TextLayerTensors(
        p_input_layernorm_weight=_bind_tensor(
            bindings,
            'p_input_layernorm_weight',
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
            bindings,
            'p_post_attention_layernorm_weight',
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.post_attention_layernorm.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(1024,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_post_attention_layernorm_weight' in request_state_outputs,
            ),
        ),
        p_attn_q_proj_weight=_bind_tensor(
            bindings,
            'p_attn_q_proj_weight',
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
            bindings,
            'p_attn_k_proj_weight',
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
            bindings,
            'p_attn_v_proj_weight',
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
            bindings,
            'p_attn_o_proj_weight',
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
            bindings,
            'p_attn_q_norm_weight',
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
            bindings,
            'p_attn_k_norm_weight',
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
            bindings,
            'p_mlp_gate_proj_weight',
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
            bindings,
            'p_mlp_up_proj_weight',
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
            bindings,
            'p_mlp_down_proj_weight',
            _declare_tensor(
            name=f"thinker.model.layers.{layer_idx}.mlp.down_proj.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(1024, 3072)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_mlp_down_proj_weight' in request_state_outputs,
            ),
        ),
        hidden_states=_bind_tensor(
            bindings,
            'hidden_states',
            _declare_tensor(
            name=f"{prefix}.hidden_states",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='hidden_states' in request_state_outputs,
            ),
        ),
        cache_position=_bind_tensor(
            bindings,
            'cache_position',
            _declare_tensor(
            name=f"{prefix}.cache_position",
            spec=TensorSpec(dtype='int32', shape=(151,)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='cache_position' in request_state_outputs,
            ),
        ),
        position_embeddings_0=_bind_tensor(
            bindings,
            'position_embeddings_0',
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
            bindings,
            'position_embeddings_1',
            _declare_tensor(
            name=f"{prefix}.position_embeddings_1",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 128)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='position_embeddings_1' in request_state_outputs,
            ),
        ),
        to=_bind_tensor(
            bindings,
            'to',
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
            bindings,
            'pow_1',
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
            bindings,
            'mean',
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
            bindings,
            'add',
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
            bindings,
            'rsqrt',
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
            bindings,
            'mul',
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
            bindings,
            'to_1',
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
            bindings,
            'mul_1',
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
            bindings,
            'linear',
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
            bindings,
            'view',
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
            bindings,
            'to_2',
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
            bindings,
            'pow_2',
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
            bindings,
            'mean_1',
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
            bindings,
            'add_1',
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
            bindings,
            'rsqrt_1',
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
            bindings,
            'mul_2',
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
            bindings,
            'to_3',
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
            bindings,
            'mul_3',
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
            bindings,
            'transpose',
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
            bindings,
            'linear_1',
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
            bindings,
            'view_1',
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
            bindings,
            'to_4',
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
            bindings,
            'pow_3',
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
            bindings,
            'mean_2',
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
            bindings,
            'add_2',
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
            bindings,
            'rsqrt_2',
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
            bindings,
            'mul_4',
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
            bindings,
            'to_5',
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
            bindings,
            'mul_5',
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
            bindings,
            'transpose_1',
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
            bindings,
            'linear_2',
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
            bindings,
            'view_2',
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
            bindings,
            'transpose_2',
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
            bindings,
            'unsqueeze',
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
            bindings,
            'unsqueeze_1',
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
            bindings,
            'mul_6',
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
            bindings,
            'slice_1',
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
            bindings,
            'slice_2',
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
            bindings,
            'neg',
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
            bindings,
            'cat',
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
            bindings,
            'mul_7',
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
            bindings,
            'add_3',
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
            bindings,
            'mul_8',
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
            bindings,
            'slice_3',
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
            bindings,
            'slice_4',
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
            bindings,
            'neg_1',
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
            bindings,
            'cat_1',
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
            bindings,
            'mul_9',
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
            bindings,
            'add_4',
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
            bindings,
            'index_copy',
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
            bindings,
            'index_copy_1',
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
            bindings,
            'scaled_dot_product_attention',
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
            bindings,
            'transpose_3',
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
            bindings,
            'contiguous',
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
            bindings,
            'reshape',
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
            bindings,
            'linear_3',
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
            bindings,
            'add_5',
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
            bindings,
            'to_6',
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
            bindings,
            'pow_4',
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
            bindings,
            'mean_3',
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
            bindings,
            'add_6',
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
            bindings,
            'rsqrt_3',
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
            bindings,
            'mul_10',
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
            bindings,
            'to_7',
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
            bindings,
            'mul_11',
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
            bindings,
            'linear_4',
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
            bindings,
            'silu',
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
            bindings,
            'linear_5',
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
            bindings,
            'mul_12',
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
            bindings,
            'linear_6',
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
            bindings,
            'add_7',
            _declare_tensor(
            name=f"{prefix}.add_7",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add_7' in request_state_outputs,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
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
    bindings: Mapping[str, LogicalTensor] | None,
    field: str,
    tensor: LogicalTensor,
) -> LogicalTensor:
    if bindings is None:
        return tensor
    bound = bindings.get(field)
    if bound is None:
        return tensor
    if bound.spec != tensor.spec:
        raise ValueError(f"{field} binding spec {bound.spec} does not match {tensor.spec}")
    return bound


def _validate_bindings(
    bindings: Mapping[str, LogicalTensor] | None,
    tensor_names: frozenset[str],
) -> None:
    if bindings is None:
        return
    unknown = frozenset(bindings) - tensor_names
    if unknown:
        raise ValueError(f"unknown tensor bindings: {sorted(unknown)}")


def _validate_request_state_outputs(
    request_state_outputs: Collection[str],
    output_names: frozenset[str],
) -> None:
    unknown = frozenset(request_state_outputs) - output_names
    if unknown:
        raise ValueError(f"request_state_outputs must name module outputs, got {sorted(unknown)}")
