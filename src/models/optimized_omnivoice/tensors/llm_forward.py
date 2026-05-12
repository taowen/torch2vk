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
class LlmLayerTensors:
    p_layers_0_self_attn_q_proj_weight: LogicalTensor
    p_layers_0_self_attn_k_proj_weight: LogicalTensor
    p_layers_0_self_attn_v_proj_weight: LogicalTensor
    p_layers_0_self_attn_o_proj_weight: LogicalTensor
    p_layers_0_self_attn_q_norm_weight: LogicalTensor
    p_layers_0_self_attn_k_norm_weight: LogicalTensor
    p_layers_0_mlp_gate_proj_weight: LogicalTensor
    p_layers_0_mlp_up_proj_weight: LogicalTensor
    p_layers_0_mlp_down_proj_weight: LogicalTensor
    p_layers_0_input_layernorm_weight: LogicalTensor
    p_layers_0_post_attention_layernorm_weight: LogicalTensor
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


LLM_LAYER_OUTPUT: str = 'add_7'


def create_llm_layer(
    prefix: str,
    layer_idx: int,
    *,
    p_layers_0_self_attn_q_proj_weight: LogicalTensor | None = None,
    p_layers_0_self_attn_k_proj_weight: LogicalTensor | None = None,
    p_layers_0_self_attn_v_proj_weight: LogicalTensor | None = None,
    p_layers_0_self_attn_o_proj_weight: LogicalTensor | None = None,
    p_layers_0_self_attn_q_norm_weight: LogicalTensor | None = None,
    p_layers_0_self_attn_k_norm_weight: LogicalTensor | None = None,
    p_layers_0_mlp_gate_proj_weight: LogicalTensor | None = None,
    p_layers_0_mlp_up_proj_weight: LogicalTensor | None = None,
    p_layers_0_mlp_down_proj_weight: LogicalTensor | None = None,
    p_layers_0_input_layernorm_weight: LogicalTensor | None = None,
    p_layers_0_post_attention_layernorm_weight: LogicalTensor | None = None,
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
) -> LlmLayerTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('add_7',)))
    tensors = LlmLayerTensors(
        p_layers_0_self_attn_q_proj_weight=_bind_tensor(
            p_layers_0_self_attn_q_proj_weight,
            _declare_tensor(
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.q_proj.weight",
                reference_key=None,
                spec=_quantized_weight_spec(f"llm.layers.{layer_idx}.self_attn.q_proj.weight", dtype='float32', shape=(2048, 1024)),
                layout=_quantized_weight_layout(f"llm.layers.{layer_idx}.self_attn.q_proj.weight", dtype='float32', shape=(2048, 1024)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_q_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_self_attn_k_proj_weight=_bind_tensor(
            p_layers_0_self_attn_k_proj_weight,
            _declare_tensor(
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.k_proj.weight",
                reference_key=None,
                spec=_quantized_weight_spec(f"llm.layers.{layer_idx}.self_attn.k_proj.weight", dtype='float32', shape=(1024, 1024)),
                layout=_quantized_weight_layout(f"llm.layers.{layer_idx}.self_attn.k_proj.weight", dtype='float32', shape=(1024, 1024)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_k_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_self_attn_v_proj_weight=_bind_tensor(
            p_layers_0_self_attn_v_proj_weight,
            _declare_tensor(
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.v_proj.weight",
                reference_key=None,
                spec=_quantized_weight_spec(f"llm.layers.{layer_idx}.self_attn.v_proj.weight", dtype='float32', shape=(1024, 1024)),
                layout=_quantized_weight_layout(f"llm.layers.{layer_idx}.self_attn.v_proj.weight", dtype='float32', shape=(1024, 1024)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_v_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_self_attn_o_proj_weight=_bind_tensor(
            p_layers_0_self_attn_o_proj_weight,
            _declare_tensor(
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.o_proj.weight",
                reference_key=None,
                spec=_quantized_weight_spec(f"llm.layers.{layer_idx}.self_attn.o_proj.weight", dtype='float32', shape=(1024, 2048)),
                layout=_quantized_weight_layout(f"llm.layers.{layer_idx}.self_attn.o_proj.weight", dtype='float32', shape=(1024, 2048)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_o_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_self_attn_q_norm_weight=_bind_tensor(
            p_layers_0_self_attn_q_norm_weight,
            _declare_tensor(
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.q_norm.weight",
                reference_key=None,
                spec=_quantized_weight_spec(f"llm.layers.{layer_idx}.self_attn.q_norm.weight", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout(f"llm.layers.{layer_idx}.self_attn.q_norm.weight", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_q_norm_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_self_attn_k_norm_weight=_bind_tensor(
            p_layers_0_self_attn_k_norm_weight,
            _declare_tensor(
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.k_norm.weight",
                reference_key=None,
                spec=_quantized_weight_spec(f"llm.layers.{layer_idx}.self_attn.k_norm.weight", dtype='float32', shape=(128,)),
                layout=_quantized_weight_layout(f"llm.layers.{layer_idx}.self_attn.k_norm.weight", dtype='float32', shape=(128,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_k_norm_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_mlp_gate_proj_weight=_bind_tensor(
            p_layers_0_mlp_gate_proj_weight,
            _declare_tensor(
                checkpoint_key=f"llm.layers.{layer_idx}.mlp.gate_proj.weight",
                reference_key=None,
                spec=_quantized_weight_spec(f"llm.layers.{layer_idx}.mlp.gate_proj.weight", dtype='float32', shape=(3072, 1024)),
                layout=_quantized_weight_layout(f"llm.layers.{layer_idx}.mlp.gate_proj.weight", dtype='float32', shape=(3072, 1024)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_mlp_gate_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_mlp_up_proj_weight=_bind_tensor(
            p_layers_0_mlp_up_proj_weight,
            _declare_tensor(
                checkpoint_key=f"llm.layers.{layer_idx}.mlp.up_proj.weight",
                reference_key=None,
                spec=_quantized_weight_spec(f"llm.layers.{layer_idx}.mlp.up_proj.weight", dtype='float32', shape=(3072, 1024)),
                layout=_quantized_weight_layout(f"llm.layers.{layer_idx}.mlp.up_proj.weight", dtype='float32', shape=(3072, 1024)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_mlp_up_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_mlp_down_proj_weight=_bind_tensor(
            p_layers_0_mlp_down_proj_weight,
            _declare_tensor(
                checkpoint_key=f"llm.layers.{layer_idx}.mlp.down_proj.weight",
                reference_key=None,
                spec=_quantized_weight_spec(f"llm.layers.{layer_idx}.mlp.down_proj.weight", dtype='float32', shape=(1024, 3072)),
                layout=_quantized_weight_layout(f"llm.layers.{layer_idx}.mlp.down_proj.weight", dtype='float32', shape=(1024, 3072)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_mlp_down_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_input_layernorm_weight=_bind_tensor(
            p_layers_0_input_layernorm_weight,
            _declare_tensor(
                checkpoint_key=f"llm.layers.{layer_idx}.input_layernorm.weight",
                reference_key=None,
                spec=_quantized_weight_spec(f"llm.layers.{layer_idx}.input_layernorm.weight", dtype='float32', shape=(1024,)),
                layout=_quantized_weight_layout(f"llm.layers.{layer_idx}.input_layernorm.weight", dtype='float32', shape=(1024,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_input_layernorm_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_post_attention_layernorm_weight=_bind_tensor(
            p_layers_0_post_attention_layernorm_weight,
            _declare_tensor(
                checkpoint_key=f"llm.layers.{layer_idx}.post_attention_layernorm.weight",
                reference_key=None,
                spec=_quantized_weight_spec(f"llm.layers.{layer_idx}.post_attention_layernorm.weight", dtype='float32', shape=(1024,)),
                layout=_quantized_weight_layout(f"llm.layers.{layer_idx}.post_attention_layernorm.weight", dtype='float32', shape=(1024,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_post_attention_layernorm_weight' in request_state_outputs,
            ),
        ),
        to=_bind_tensor(
            to,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='to',
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 2048)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 16, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 16, 128)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 16, 128)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 16, 1)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 16, 1)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 16, 1)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 16, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 16, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 16, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 8, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 8, 128)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 8, 128)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 8, 1)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 8, 1)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 8, 1)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 8, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 8, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 8, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 8, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 1, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 1, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 64)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 64)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 64)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 64)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 64)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 64)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_4' in request_state_outputs,
            ),
        ),
        scaled_dot_product_attention=_bind_tensor(
            scaled_dot_product_attention,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='scaled_dot_product_attention',
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 16, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='transpose_3' in request_state_outputs,
            ),
        ),
        contiguous=_bind_tensor(
            contiguous,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='contiguous',
                spec=TensorSpec(dtype='float16', shape=(2, 85, 16, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='contiguous' in request_state_outputs,
            ),
        ),
        reshape=_bind_tensor(
            reshape,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='reshape',
                spec=TensorSpec(dtype='float16', shape=(2, 85, 2048)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1)),
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
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 3072)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 3072)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 3072)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 3072)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_7' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    return tensors


@dataclass(frozen=True, slots=True)
class LlmForwardTensors:
    p_norm_weight: LogicalTensor
    hidden_states: LogicalTensor
    cos: LogicalTensor
    sin: LogicalTensor
    attention_mask: LogicalTensor
    to_224: LogicalTensor
    pow_113: LogicalTensor
    mean_112: LogicalTensor
    add_224: LogicalTensor
    rsqrt_112: LogicalTensor
    mul_364: LogicalTensor
    to_225: LogicalTensor
    mul_365: LogicalTensor
    layers: list[LlmLayerTensors]


LLM_FORWARD_OUTPUT: str = 'mul_365'


def create_llm_forward(
    prefix: str,
    *,
    p_norm_weight: LogicalTensor | None = None,
    hidden_states: LogicalTensor | None = None,
    cos: LogicalTensor | None = None,
    sin: LogicalTensor | None = None,
    attention_mask: LogicalTensor | None = None,
    to_224: LogicalTensor | None = None,
    pow_113: LogicalTensor | None = None,
    mean_112: LogicalTensor | None = None,
    add_224: LogicalTensor | None = None,
    rsqrt_112: LogicalTensor | None = None,
    mul_364: LogicalTensor | None = None,
    to_225: LogicalTensor | None = None,
    mul_365: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> LlmForwardTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('mul_365',)))
    tensors = LlmForwardTensors(
        p_norm_weight=_bind_tensor(
            p_norm_weight,
            _declare_tensor(
                checkpoint_key="llm.norm.weight",
                reference_key=None,
                spec=_quantized_weight_spec("llm.norm.weight", dtype='float32', shape=(1024,)),
                layout=_quantized_weight_layout("llm.norm.weight", dtype='float32', shape=(1024,)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_norm_weight' in request_state_outputs,
            ),
        ),
        hidden_states=_bind_tensor(
            hidden_states,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='hidden_states' in request_state_outputs,
            ),
        ),
        cos=_bind_tensor(
            cos,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float16', shape=(2, 85, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='cos' in request_state_outputs,
            ),
        ),
        sin=_bind_tensor(
            sin,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float16', shape=(2, 85, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='sin' in request_state_outputs,
            ),
        ),
        attention_mask=_bind_tensor(
            attention_mask,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float16', shape=(2, 1, 85, 85)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='attention_mask' in request_state_outputs,
            ),
        ),
        to_224=_bind_tensor(
            to_224,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='to_224',
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_224' in request_state_outputs,
            ),
        ),
        pow_113=_bind_tensor(
            pow_113,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='pow_113',
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='pow_113' in request_state_outputs,
            ),
        ),
        mean_112=_bind_tensor(
            mean_112,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mean_112',
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mean_112' in request_state_outputs,
            ),
        ),
        add_224=_bind_tensor(
            add_224,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add_224',
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_224' in request_state_outputs,
            ),
        ),
        rsqrt_112=_bind_tensor(
            rsqrt_112,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='rsqrt_112',
                spec=TensorSpec(dtype='float32', shape=(2, 85, 1)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rsqrt_112' in request_state_outputs,
            ),
        ),
        mul_364=_bind_tensor(
            mul_364,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_364',
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_364' in request_state_outputs,
            ),
        ),
        to_225=_bind_tensor(
            to_225,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='to_225',
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to_225' in request_state_outputs,
            ),
        ),
        mul_365=_bind_tensor(
            mul_365,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='mul_365',
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul_365' in request_state_outputs,
            ),
        ),
        layers=[create_llm_layer(prefix, layer_idx=i) for i in range(28)],
    )
    bind_logical_tensor_names(tensors, prefix)
    _alias_carry = tensors.hidden_states
    for layer_t in tensors.layers:
        _bind_alias_source(_alias_carry, layer_t.to)
        _bind_alias_source(layer_t.mul, layer_t.to_1)
        _bind_alias_source(layer_t.linear, layer_t.view)
        _bind_alias_source(layer_t.view, layer_t.to_2)
        _bind_alias_source(layer_t.mul_2, layer_t.to_3)
        _bind_alias_source(layer_t.linear_1, layer_t.view_1)
        _bind_alias_source(layer_t.view_1, layer_t.to_4)
        _bind_alias_source(layer_t.mul_4, layer_t.to_5)
        _bind_alias_source(layer_t.linear_2, layer_t.view_2)
        _bind_alias_source(tensors.cos, layer_t.unsqueeze)
        _bind_alias_source(tensors.sin, layer_t.unsqueeze_1)
        _bind_alias_source(layer_t.transpose_3, layer_t.contiguous)
        _bind_alias_source(layer_t.contiguous, layer_t.reshape)
        _bind_alias_source(layer_t.add_5, layer_t.to_6)
        _bind_alias_source(layer_t.mul_10, layer_t.to_7)
        _alias_carry = layer_t.add_7
    _bind_alias_source(_alias_carry, tensors.to_224)
    _bind_alias_source(tensors.mul_364, tensors.to_225)
    return tensors


_Q6_TENSOR_NAMES = frozenset(('audio_heads.weight', 'llm.layers.0.mlp.down_proj.weight', 'llm.layers.0.self_attn.v_proj.weight', 'llm.layers.1.mlp.down_proj.weight', 'llm.layers.1.self_attn.v_proj.weight', 'llm.layers.11.mlp.down_proj.weight', 'llm.layers.11.self_attn.v_proj.weight', 'llm.layers.14.mlp.down_proj.weight', 'llm.layers.14.self_attn.v_proj.weight', 'llm.layers.17.mlp.down_proj.weight', 'llm.layers.17.self_attn.v_proj.weight', 'llm.layers.2.mlp.down_proj.weight', 'llm.layers.2.self_attn.v_proj.weight', 'llm.layers.20.mlp.down_proj.weight', 'llm.layers.20.self_attn.v_proj.weight', 'llm.layers.23.mlp.down_proj.weight', 'llm.layers.23.self_attn.v_proj.weight', 'llm.layers.24.mlp.down_proj.weight', 'llm.layers.24.self_attn.v_proj.weight', 'llm.layers.25.mlp.down_proj.weight', 'llm.layers.25.self_attn.v_proj.weight', 'llm.layers.26.mlp.down_proj.weight', 'llm.layers.26.self_attn.v_proj.weight', 'llm.layers.27.mlp.down_proj.weight', 'llm.layers.27.self_attn.v_proj.weight', 'llm.layers.5.mlp.down_proj.weight', 'llm.layers.5.self_attn.v_proj.weight', 'llm.layers.8.mlp.down_proj.weight', 'llm.layers.8.self_attn.v_proj.weight'))
_Q6_TENSOR_PREFIXES = ()
_Q8_TENSOR_NAMES = frozenset(('audio_embeddings.weight', 'llm.embed_tokens.weight'))
_Q8_TENSOR_PREFIXES = ()


def _quantized_weight_spec(checkpoint_key: str, *, dtype: str, shape: tuple[int, ...]) -> TensorSpec:
    if dtype not in ("float32", "float16", "bfloat16"):
        return TensorSpec(dtype=dtype, shape=shape)
    force_q6 = checkpoint_key in _Q6_TENSOR_NAMES or checkpoint_key.startswith(_Q6_TENSOR_PREFIXES)
    force_q8 = checkpoint_key in _Q8_TENSOR_NAMES or checkpoint_key.startswith(_Q8_TENSOR_PREFIXES)
    if force_q6 and len(shape) >= 2:
        n, k = _quantized_matrix_shape(shape)
        if k % 256 != 0:
            raise ValueError(f"Q6_K tensor {checkpoint_key} requires K to be divisible by 256, got {k}")
        return TensorSpec(dtype="uint16", shape=(n, k // 256 * 105))
    if force_q8 and len(shape) >= 2:
        n, k = _quantized_matrix_shape(shape)
        padded_k = _round_up(k, 32)
        return TensorSpec(dtype="uint16", shape=(n, padded_k // 32 * 17))
    if len(shape) != 2:
        return TensorSpec(dtype=dtype, shape=shape)
    n, k = shape
    if k % 256 != 0:
        if k % 32 != 0:
            return TensorSpec(dtype="float32", shape=shape)
        return TensorSpec(dtype="uint16", shape=(n, k // 32 * 17))
    return TensorSpec(dtype="uint32", shape=(n, k // 256 * 36))


def _quantized_weight_layout(checkpoint_key: str, *, dtype: str, shape: tuple[int, ...]) -> TensorLayout:
    if dtype not in ("float32", "float16", "bfloat16"):
        return CONTIGUOUS_LAYOUT
    force_q6 = checkpoint_key in _Q6_TENSOR_NAMES or checkpoint_key.startswith(_Q6_TENSOR_PREFIXES)
    force_q8 = checkpoint_key in _Q8_TENSOR_NAMES or checkpoint_key.startswith(_Q8_TENSOR_PREFIXES)
    if force_q6 and len(shape) >= 2:
        _, k = _quantized_matrix_shape(shape)
        if k % 256 != 0:
            raise ValueError(f"Q6_K tensor {checkpoint_key} requires K to be divisible by 256, got {k}")
        return q6_k_halfwords_layout(logical_k=k)
    if force_q8 and len(shape) >= 2:
        _, k = _quantized_matrix_shape(shape)
        return q8_0_halfwords_layout(logical_k=k)
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
