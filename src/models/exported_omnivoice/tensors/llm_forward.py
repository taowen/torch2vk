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
    scaled_dot_product_attention: LogicalTensor
    transpose_3: LogicalTensor
    contiguous: LogicalTensor
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


LLM_LAYER_OUTPUT: str = 'add_3'


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
    scaled_dot_product_attention: LogicalTensor | None = None,
    transpose_3: LogicalTensor | None = None,
    contiguous: LogicalTensor | None = None,
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
) -> LlmLayerTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('add_3',)))
    tensors = LlmLayerTensors(
        p_layers_0_self_attn_q_proj_weight=_bind_tensor(
            p_layers_0_self_attn_q_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.q_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(2048, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_q_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_self_attn_k_proj_weight=_bind_tensor(
            p_layers_0_self_attn_k_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.k_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1024, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_k_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_self_attn_v_proj_weight=_bind_tensor(
            p_layers_0_self_attn_v_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.v_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1024, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_v_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_self_attn_o_proj_weight=_bind_tensor(
            p_layers_0_self_attn_o_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.o_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1024, 2048)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_o_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_self_attn_q_norm_weight=_bind_tensor(
            p_layers_0_self_attn_q_norm_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.q_norm.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(128,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_q_norm_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_self_attn_k_norm_weight=_bind_tensor(
            p_layers_0_self_attn_k_norm_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"llm.layers.{layer_idx}.self_attn.k_norm.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(128,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_self_attn_k_norm_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_mlp_gate_proj_weight=_bind_tensor(
            p_layers_0_mlp_gate_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"llm.layers.{layer_idx}.mlp.gate_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(3072, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_mlp_gate_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_mlp_up_proj_weight=_bind_tensor(
            p_layers_0_mlp_up_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"llm.layers.{layer_idx}.mlp.up_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(3072, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_mlp_up_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_mlp_down_proj_weight=_bind_tensor(
            p_layers_0_mlp_down_proj_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"llm.layers.{layer_idx}.mlp.down_proj.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1024, 3072)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_mlp_down_proj_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_input_layernorm_weight=_bind_tensor(
            p_layers_0_input_layernorm_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"llm.layers.{layer_idx}.input_layernorm.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1024,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_input_layernorm_weight' in request_state_outputs,
            ),
        ),
        p_layers_0_post_attention_layernorm_weight=_bind_tensor(
            p_layers_0_post_attention_layernorm_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=f"llm.layers.{layer_idx}.post_attention_layernorm.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1024,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_layers_0_post_attention_layernorm_weight' in request_state_outputs,
            ),
        ),
        rms_norm=_bind_tensor(
            rms_norm,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rms_norm',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='view',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(2, 85, 16, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 16, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_1',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='view_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(2, 85, 8, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 8, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_2',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='view_2',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='transpose_2',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(2, 1, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='slice_2',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 16, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 128)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='slice_4',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='neg_1',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='cat_1',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 128)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 8, 85, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_1' in request_state_outputs,
            ),
        ),
        scaled_dot_product_attention=_bind_tensor(
            scaled_dot_product_attention,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='scaled_dot_product_attention',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='transpose_3',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='contiguous',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='reshape',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_3',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='silu',
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_5',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(2, 85, 3072)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 3072)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
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
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_3' in request_state_outputs,
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
    rms_norm_112: LogicalTensor
    layers: list[LlmLayerTensors]


LLM_FORWARD_OUTPUT: str = 'rms_norm_112'


def create_llm_forward(
    prefix: str,
    *,
    p_norm_weight: LogicalTensor | None = None,
    hidden_states: LogicalTensor | None = None,
    cos: LogicalTensor | None = None,
    sin: LogicalTensor | None = None,
    attention_mask: LogicalTensor | None = None,
    rms_norm_112: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> LlmForwardTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('rms_norm_112',)))
    tensors = LlmForwardTensors(
        p_norm_weight=_bind_tensor(
            p_norm_weight,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key="llm.norm.weight",
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1024,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_norm_weight' in request_state_outputs,
            ),
        ),
        hidden_states=_bind_tensor(
            hidden_states,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
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
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(2, 1, 85, 85)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='attention_mask' in request_state_outputs,
            ),
        ),
        rms_norm_112=_bind_tensor(
            rms_norm_112,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='rms_norm_112',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(2, 85, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='rms_norm_112' in request_state_outputs,
            ),
        ),
        layers=[create_llm_layer(prefix, layer_idx=i) for i in range(28)],
    )
    bind_logical_tensor_names(tensors, prefix)
    _alias_carry = tensors.hidden_states
    for layer_t in tensors.layers:
        _bind_alias_source(layer_t.linear, layer_t.view)
        _bind_alias_source(layer_t.linear_1, layer_t.view_1)
        _bind_alias_source(layer_t.linear_2, layer_t.view_2)
        _bind_alias_source(tensors.cos, layer_t.unsqueeze)
        _bind_alias_source(tensors.sin, layer_t.unsqueeze_1)
        _bind_alias_source(layer_t.transpose_3, layer_t.contiguous)
        _bind_alias_source(layer_t.contiguous, layer_t.reshape)
        _alias_carry = layer_t.add_3
    return tensors


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
