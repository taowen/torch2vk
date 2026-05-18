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
    rms_norm: LogicalTensor
    linear: LogicalTensor
    view: LogicalTensor
    rms_norm_1: LogicalTensor
    transpose: LogicalTensor
    linear_1: LogicalTensor
    view_1: LogicalTensor
    rms_norm_2: LogicalTensor
    transpose_1: LogicalTensor
    linear_2: LogicalTensor
    view_2: LogicalTensor
    transpose_2: LogicalTensor
    unsqueeze: LogicalTensor
    unsqueeze_1: LogicalTensor
    mul: LogicalTensor
    slice_1: LogicalTensor
    slice_2: LogicalTensor
    neg: LogicalTensor
    cat: LogicalTensor
    mul_1: LogicalTensor
    add: LogicalTensor
    mul_2: LogicalTensor
    slice_3: LogicalTensor
    slice_4: LogicalTensor
    neg_1: LogicalTensor
    cat_1: LogicalTensor
    mul_3: LogicalTensor
    add_1: LogicalTensor
    index_copy: LogicalTensor
    index_copy_1: LogicalTensor
    scaled_dot_product_attention: LogicalTensor
    transpose_3: LogicalTensor
    reshape: LogicalTensor
    linear_3: LogicalTensor
    add_2: LogicalTensor
    rms_norm_3: LogicalTensor
    linear_4: LogicalTensor
    silu: LogicalTensor
    linear_5: LogicalTensor
    mul_4: LogicalTensor
    linear_6: LogicalTensor
    add_3: LogicalTensor


DECODE_LAYER_OUTPUT: str = 'add_3'


def create_decode_layer(
    prefix: str,
    layer_idx: int,
    *,
    max_sequence_length: int,
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
    rms_norm: LogicalTensor | None = None,
    linear: LogicalTensor | None = None,
    view: LogicalTensor | None = None,
    rms_norm_1: LogicalTensor | None = None,
    transpose: LogicalTensor | None = None,
    linear_1: LogicalTensor | None = None,
    view_1: LogicalTensor | None = None,
    rms_norm_2: LogicalTensor | None = None,
    transpose_1: LogicalTensor | None = None,
    linear_2: LogicalTensor | None = None,
    view_2: LogicalTensor | None = None,
    transpose_2: LogicalTensor | None = None,
    unsqueeze: LogicalTensor | None = None,
    unsqueeze_1: LogicalTensor | None = None,
    mul: LogicalTensor | None = None,
    slice_1: LogicalTensor | None = None,
    slice_2: LogicalTensor | None = None,
    neg: LogicalTensor | None = None,
    cat: LogicalTensor | None = None,
    mul_1: LogicalTensor | None = None,
    add: LogicalTensor | None = None,
    mul_2: LogicalTensor | None = None,
    slice_3: LogicalTensor | None = None,
    slice_4: LogicalTensor | None = None,
    neg_1: LogicalTensor | None = None,
    cat_1: LogicalTensor | None = None,
    mul_3: LogicalTensor | None = None,
    add_1: LogicalTensor | None = None,
    index_copy: LogicalTensor | None = None,
    index_copy_1: LogicalTensor | None = None,
    scaled_dot_product_attention: LogicalTensor | None = None,
    transpose_3: LogicalTensor | None = None,
    reshape: LogicalTensor | None = None,
    linear_3: LogicalTensor | None = None,
    add_2: LogicalTensor | None = None,
    rms_norm_3: LogicalTensor | None = None,
    linear_4: LogicalTensor | None = None,
    silu: LogicalTensor | None = None,
    linear_5: LogicalTensor | None = None,
    mul_4: LogicalTensor | None = None,
    linear_6: LogicalTensor | None = None,
    add_3: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> DecodeLayerTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('add_3', 'index_copy', 'index_copy_1')))
    tensors = DecodeLayerTensors(
        p_attn_q_proj_weight=_bind_tensor(
            p_attn_q_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.q_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"thinker.model.layers.{layer_idx}.self_attn.q_proj.weight", dtype='float32', shape=(2048, 1024)),
                layout=_quantized_weight_layout(f"thinker.model.layers.{layer_idx}.self_attn.q_proj.weight", dtype='float32', shape=(2048, 1024)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_q_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_k_proj_weight=_bind_tensor(
            p_attn_k_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.k_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"thinker.model.layers.{layer_idx}.self_attn.k_proj.weight", dtype='float32', shape=(1024, 1024)),
                layout=_quantized_weight_layout(f"thinker.model.layers.{layer_idx}.self_attn.k_proj.weight", dtype='float32', shape=(1024, 1024)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_k_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_v_proj_weight=_bind_tensor(
            p_attn_v_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.v_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"thinker.model.layers.{layer_idx}.self_attn.v_proj.weight", dtype='float32', shape=(1024, 1024)),
                layout=_quantized_weight_layout(f"thinker.model.layers.{layer_idx}.self_attn.v_proj.weight", dtype='float32', shape=(1024, 1024)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_v_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_o_proj_weight=_bind_tensor(
            p_attn_o_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.o_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"thinker.model.layers.{layer_idx}.self_attn.o_proj.weight", dtype='float32', shape=(1024, 2048)),
                layout=_quantized_weight_layout(f"thinker.model.layers.{layer_idx}.self_attn.o_proj.weight", dtype='float32', shape=(1024, 2048)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_o_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_q_norm_weight=_bind_tensor(
            p_attn_q_norm_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.q_norm.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"thinker.model.layers.{layer_idx}.self_attn.q_norm.weight", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout(f"thinker.model.layers.{layer_idx}.self_attn.q_norm.weight", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_q_norm_weight' in request_state_outputs,
            ),
        ),
        p_attn_k_norm_weight=_bind_tensor(
            p_attn_k_norm_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"thinker.model.layers.{layer_idx}.self_attn.k_norm.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"thinker.model.layers.{layer_idx}.self_attn.k_norm.weight", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout(f"thinker.model.layers.{layer_idx}.self_attn.k_norm.weight", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_attn_k_norm_weight' in request_state_outputs,
            ),
        ),
        p_mlp_gate_proj_weight=_bind_tensor(
            p_mlp_gate_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"thinker.model.layers.{layer_idx}.mlp.gate_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"thinker.model.layers.{layer_idx}.mlp.gate_proj.weight", dtype='float32', shape=(3072, 1024)),
                layout=_quantized_weight_layout(f"thinker.model.layers.{layer_idx}.mlp.gate_proj.weight", dtype='float32', shape=(3072, 1024)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_mlp_gate_proj_weight' in request_state_outputs,
            ),
        ),
        p_mlp_up_proj_weight=_bind_tensor(
            p_mlp_up_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"thinker.model.layers.{layer_idx}.mlp.up_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"thinker.model.layers.{layer_idx}.mlp.up_proj.weight", dtype='float32', shape=(3072, 1024)),
                layout=_quantized_weight_layout(f"thinker.model.layers.{layer_idx}.mlp.up_proj.weight", dtype='float32', shape=(3072, 1024)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_mlp_up_proj_weight' in request_state_outputs,
            ),
        ),
        p_mlp_down_proj_weight=_bind_tensor(
            p_mlp_down_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"thinker.model.layers.{layer_idx}.mlp.down_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"thinker.model.layers.{layer_idx}.mlp.down_proj.weight", dtype='float32', shape=(1024, 3072)),
                layout=_quantized_weight_layout(f"thinker.model.layers.{layer_idx}.mlp.down_proj.weight", dtype='float32', shape=(1024, 3072)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_mlp_down_proj_weight' in request_state_outputs,
            ),
        ),
        p_input_layernorm_weight=_bind_tensor(
            p_input_layernorm_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"thinker.model.layers.{layer_idx}.input_layernorm.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"thinker.model.layers.{layer_idx}.input_layernorm.weight", dtype='float32', shape=(1024,)),
                layout=_quantized_weight_layout(f"thinker.model.layers.{layer_idx}.input_layernorm.weight", dtype='float32', shape=(1024,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_input_layernorm_weight' in request_state_outputs,
            ),
        ),
        p_post_attention_layernorm_weight=_bind_tensor(
            p_post_attention_layernorm_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"thinker.model.layers.{layer_idx}.post_attention_layernorm.weight",
                reference_key=None,
                layer=prefix,
                spec=_quantized_weight_spec(f"thinker.model.layers.{layer_idx}.post_attention_layernorm.weight", dtype='float32', shape=(1024,)),
                layout=_quantized_weight_layout(f"thinker.model.layers.{layer_idx}.post_attention_layernorm.weight", dtype='float32', shape=(1024,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_post_attention_layernorm_weight' in request_state_outputs,
            ),
        ),
        hidden_states=_bind_tensor(
            hidden_states,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='position_embeddings_1' in request_state_outputs,
            ),
        ),
        rms_norm=_bind_tensor(
            rms_norm,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rms_norm',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rms_norm' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            linear,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 2048)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='view',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 16, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='view' in request_state_outputs,
            ),
        ),
        rms_norm_1=_bind_tensor(
            rms_norm_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rms_norm_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 16, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rms_norm_1' in request_state_outputs,
            ),
        ),
        transpose=_bind_tensor(
            transpose,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='transpose',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 16, 1, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='view_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 8, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='view_1' in request_state_outputs,
            ),
        ),
        rms_norm_2=_bind_tensor(
            rms_norm_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rms_norm_2',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 8, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rms_norm_2' in request_state_outputs,
            ),
        ),
        transpose_1=_bind_tensor(
            transpose_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='transpose_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 8, 1, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_2',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='view_2',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 8, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='transpose_2',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 8, 1, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze_1' in request_state_outputs,
            ),
        ),
        mul=_bind_tensor(
            mul,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 16, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul' in request_state_outputs,
            ),
        ),
        slice_1=_bind_tensor(
            slice_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='slice_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 16, 1, 64)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='slice_2',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 16, 1, 64)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 16, 1, 64)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 16, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat' in request_state_outputs,
            ),
        ),
        mul_1=_bind_tensor(
            mul_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 16, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_1' in request_state_outputs,
            ),
        ),
        add=_bind_tensor(
            add,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 16, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add' in request_state_outputs,
            ),
        ),
        mul_2=_bind_tensor(
            mul_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_2',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 8, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_2' in request_state_outputs,
            ),
        ),
        slice_3=_bind_tensor(
            slice_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='slice_3',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 8, 1, 64)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='slice_4',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 8, 1, 64)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 8, 1, 64)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 8, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='cat_1' in request_state_outputs,
            ),
        ),
        mul_3=_bind_tensor(
            mul_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_3',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 8, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_3' in request_state_outputs,
            ),
        ),
        add_1=_bind_tensor(
            add_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 8, 1, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_1' in request_state_outputs,
            ),
        ),
        index_copy=_bind_tensor(
            index_copy,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='index_copy',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 8, max_sequence_length, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='index_copy_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 8, max_sequence_length, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='scaled_dot_product_attention',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 16, 1, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='transpose_3',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 16, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 2048)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_3',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_3' in request_state_outputs,
            ),
        ),
        add_2=_bind_tensor(
            add_2,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_2',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_2' in request_state_outputs,
            ),
        ),
        rms_norm_3=_bind_tensor(
            rms_norm_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rms_norm_3',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rms_norm_3' in request_state_outputs,
            ),
        ),
        linear_4=_bind_tensor(
            linear_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_4',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 3072)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='silu',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 3072)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_5',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 3072)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_5' in request_state_outputs,
            ),
        ),
        mul_4=_bind_tensor(
            mul_4,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul_4',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 3072)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_4' in request_state_outputs,
            ),
        ),
        linear_6=_bind_tensor(
            linear_6,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_6',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_6' in request_state_outputs,
            ),
        ),
        add_3=_bind_tensor(
            add_3,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_3',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_3' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    _bind_alias_source(tensors.linear, tensors.view)
    _bind_alias_source(tensors.linear_1, tensors.view_1)
    _bind_alias_source(tensors.linear_2, tensors.view_2)
    _bind_alias_source(tensors.position_embeddings_0, tensors.unsqueeze)
    _bind_alias_source(tensors.position_embeddings_1, tensors.unsqueeze_1)
    _bind_alias_source(tensors.transpose_3, tensors.reshape)
    return tensors


_F16_TENSOR_NAMES = frozenset(())
_F16_TENSOR_PREFIXES = ()
_Q6_TENSOR_NAMES = frozenset(('thinker.lm_head.weight', 'thinker.model.layers.0.mlp.down_proj.weight', 'thinker.model.layers.0.self_attn.v_proj.weight', 'thinker.model.layers.1.mlp.down_proj.weight', 'thinker.model.layers.1.self_attn.v_proj.weight', 'thinker.model.layers.11.mlp.down_proj.weight', 'thinker.model.layers.11.self_attn.v_proj.weight', 'thinker.model.layers.14.mlp.down_proj.weight', 'thinker.model.layers.14.self_attn.v_proj.weight', 'thinker.model.layers.17.mlp.down_proj.weight', 'thinker.model.layers.17.self_attn.v_proj.weight', 'thinker.model.layers.2.mlp.down_proj.weight', 'thinker.model.layers.2.self_attn.v_proj.weight', 'thinker.model.layers.20.mlp.down_proj.weight', 'thinker.model.layers.20.self_attn.v_proj.weight', 'thinker.model.layers.23.mlp.down_proj.weight', 'thinker.model.layers.23.self_attn.v_proj.weight', 'thinker.model.layers.24.mlp.down_proj.weight', 'thinker.model.layers.24.self_attn.v_proj.weight', 'thinker.model.layers.25.mlp.down_proj.weight', 'thinker.model.layers.25.self_attn.v_proj.weight', 'thinker.model.layers.26.mlp.down_proj.weight', 'thinker.model.layers.26.self_attn.v_proj.weight', 'thinker.model.layers.27.mlp.down_proj.weight', 'thinker.model.layers.27.self_attn.v_proj.weight', 'thinker.model.layers.5.mlp.down_proj.weight', 'thinker.model.layers.5.self_attn.v_proj.weight', 'thinker.model.layers.8.mlp.down_proj.weight', 'thinker.model.layers.8.self_attn.v_proj.weight'))
_Q6_TENSOR_PREFIXES = ()
_Q8_TENSOR_NAMES = frozenset(('thinker.model.embed_tokens.weight',))
_Q8_TENSOR_PREFIXES = ('thinker.audio_tower.',)


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
