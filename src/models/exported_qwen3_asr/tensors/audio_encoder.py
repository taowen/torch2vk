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
class EncoderLayerTensors:
    p_audio_tower_layers_0_self_attn_k_proj_weight: LogicalTensor
    p_audio_tower_layers_0_self_attn_k_proj_bias: LogicalTensor
    p_audio_tower_layers_0_self_attn_v_proj_weight: LogicalTensor
    p_audio_tower_layers_0_self_attn_v_proj_bias: LogicalTensor
    p_audio_tower_layers_0_self_attn_q_proj_weight: LogicalTensor
    p_audio_tower_layers_0_self_attn_q_proj_bias: LogicalTensor
    p_audio_tower_layers_0_self_attn_out_proj_weight: LogicalTensor
    p_audio_tower_layers_0_self_attn_out_proj_bias: LogicalTensor
    p_audio_tower_layers_0_self_attn_layer_norm_weight: LogicalTensor
    p_audio_tower_layers_0_self_attn_layer_norm_bias: LogicalTensor
    p_audio_tower_layers_0_fc1_weight: LogicalTensor
    p_audio_tower_layers_0_fc1_bias: LogicalTensor
    p_audio_tower_layers_0_fc2_weight: LogicalTensor
    p_audio_tower_layers_0_fc2_bias: LogicalTensor
    p_audio_tower_layers_0_final_layer_norm_weight: LogicalTensor
    p_audio_tower_layers_0_final_layer_norm_bias: LogicalTensor
    layer_norm: LogicalTensor
    linear_1: LogicalTensor
    reshape_2: LogicalTensor
    linear_2: LogicalTensor
    reshape_3: LogicalTensor
    linear_3: LogicalTensor
    reshape_4: LogicalTensor
    transpose_1: LogicalTensor
    unsqueeze: LogicalTensor
    transpose_2: LogicalTensor
    unsqueeze_1: LogicalTensor
    transpose_3: LogicalTensor
    unsqueeze_2: LogicalTensor
    scaled_dot_product_attention: LogicalTensor
    transpose_4: LogicalTensor
    contiguous: LogicalTensor
    reshape_5: LogicalTensor
    linear_4: LogicalTensor
    add_1: LogicalTensor
    layer_norm_1: LogicalTensor
    linear_5: LogicalTensor
    gelu_3: LogicalTensor
    linear_6: LogicalTensor
    add_2: LogicalTensor


ENCODER_LAYER_OUTPUT: str = 'add_2'


def create_encoder_layer(
    prefix: str,
    layer_idx: int,
    *,
    p_audio_tower_layers_0_self_attn_k_proj_weight: LogicalTensor | None = None,
    p_audio_tower_layers_0_self_attn_k_proj_bias: LogicalTensor | None = None,
    p_audio_tower_layers_0_self_attn_v_proj_weight: LogicalTensor | None = None,
    p_audio_tower_layers_0_self_attn_v_proj_bias: LogicalTensor | None = None,
    p_audio_tower_layers_0_self_attn_q_proj_weight: LogicalTensor | None = None,
    p_audio_tower_layers_0_self_attn_q_proj_bias: LogicalTensor | None = None,
    p_audio_tower_layers_0_self_attn_out_proj_weight: LogicalTensor | None = None,
    p_audio_tower_layers_0_self_attn_out_proj_bias: LogicalTensor | None = None,
    p_audio_tower_layers_0_self_attn_layer_norm_weight: LogicalTensor | None = None,
    p_audio_tower_layers_0_self_attn_layer_norm_bias: LogicalTensor | None = None,
    p_audio_tower_layers_0_fc1_weight: LogicalTensor | None = None,
    p_audio_tower_layers_0_fc1_bias: LogicalTensor | None = None,
    p_audio_tower_layers_0_fc2_weight: LogicalTensor | None = None,
    p_audio_tower_layers_0_fc2_bias: LogicalTensor | None = None,
    p_audio_tower_layers_0_final_layer_norm_weight: LogicalTensor | None = None,
    p_audio_tower_layers_0_final_layer_norm_bias: LogicalTensor | None = None,
    layer_norm: LogicalTensor | None = None,
    linear_1: LogicalTensor | None = None,
    reshape_2: LogicalTensor | None = None,
    linear_2: LogicalTensor | None = None,
    reshape_3: LogicalTensor | None = None,
    linear_3: LogicalTensor | None = None,
    reshape_4: LogicalTensor | None = None,
    transpose_1: LogicalTensor | None = None,
    unsqueeze: LogicalTensor | None = None,
    transpose_2: LogicalTensor | None = None,
    unsqueeze_1: LogicalTensor | None = None,
    transpose_3: LogicalTensor | None = None,
    unsqueeze_2: LogicalTensor | None = None,
    scaled_dot_product_attention: LogicalTensor | None = None,
    transpose_4: LogicalTensor | None = None,
    contiguous: LogicalTensor | None = None,
    reshape_5: LogicalTensor | None = None,
    linear_4: LogicalTensor | None = None,
    add_1: LogicalTensor | None = None,
    layer_norm_1: LogicalTensor | None = None,
    linear_5: LogicalTensor | None = None,
    gelu_3: LogicalTensor | None = None,
    linear_6: LogicalTensor | None = None,
    add_2: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> EncoderLayerTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('add_2',)))
    tensors = EncoderLayerTensors(
        p_audio_tower_layers_0_self_attn_k_proj_weight=_bind_tensor(
            p_audio_tower_layers_0_self_attn_k_proj_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.self_attn.k_proj.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_self_attn_k_proj_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_self_attn_k_proj_bias=_bind_tensor(
            p_audio_tower_layers_0_self_attn_k_proj_bias,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.self_attn.k_proj.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_self_attn_k_proj_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_self_attn_v_proj_weight=_bind_tensor(
            p_audio_tower_layers_0_self_attn_v_proj_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.self_attn.v_proj.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_self_attn_v_proj_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_self_attn_v_proj_bias=_bind_tensor(
            p_audio_tower_layers_0_self_attn_v_proj_bias,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.self_attn.v_proj.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_self_attn_v_proj_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_self_attn_q_proj_weight=_bind_tensor(
            p_audio_tower_layers_0_self_attn_q_proj_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.self_attn.q_proj.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_self_attn_q_proj_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_self_attn_q_proj_bias=_bind_tensor(
            p_audio_tower_layers_0_self_attn_q_proj_bias,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.self_attn.q_proj.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_self_attn_q_proj_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_self_attn_out_proj_weight=_bind_tensor(
            p_audio_tower_layers_0_self_attn_out_proj_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.self_attn.out_proj.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_self_attn_out_proj_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_self_attn_out_proj_bias=_bind_tensor(
            p_audio_tower_layers_0_self_attn_out_proj_bias,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.self_attn.out_proj.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_self_attn_out_proj_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_self_attn_layer_norm_weight=_bind_tensor(
            p_audio_tower_layers_0_self_attn_layer_norm_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.self_attn_layer_norm.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_self_attn_layer_norm_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_self_attn_layer_norm_bias=_bind_tensor(
            p_audio_tower_layers_0_self_attn_layer_norm_bias,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.self_attn_layer_norm.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_self_attn_layer_norm_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_fc1_weight=_bind_tensor(
            p_audio_tower_layers_0_fc1_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.fc1.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(3584, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_fc1_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_fc1_bias=_bind_tensor(
            p_audio_tower_layers_0_fc1_bias,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.fc1.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(3584,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_fc1_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_fc2_weight=_bind_tensor(
            p_audio_tower_layers_0_fc2_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.fc2.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896, 3584)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_fc2_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_fc2_bias=_bind_tensor(
            p_audio_tower_layers_0_fc2_bias,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.fc2.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_fc2_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_final_layer_norm_weight=_bind_tensor(
            p_audio_tower_layers_0_final_layer_norm_weight,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.final_layer_norm.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_final_layer_norm_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_layers_0_final_layer_norm_bias=_bind_tensor(
            p_audio_tower_layers_0_final_layer_norm_bias,
            _declare_tensor(
                checkpoint_key=f"thinker.audio_tower.layers.{layer_idx}.final_layer_norm.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_layers_0_final_layer_norm_bias' in request_state_outputs,
            ),
        ),
        layer_norm=_bind_tensor(
            layer_norm,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='layer_norm',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_norm' in request_state_outputs,
            ),
        ),
        linear_1=_bind_tensor(
            linear_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_1',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_1' in request_state_outputs,
            ),
        ),
        reshape_2=_bind_tensor(
            reshape_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='reshape_2',
                spec=TensorSpec(dtype='float16', shape=(133, 14, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_2' in request_state_outputs,
            ),
        ),
        linear_2=_bind_tensor(
            linear_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_2',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_2' in request_state_outputs,
            ),
        ),
        reshape_3=_bind_tensor(
            reshape_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='reshape_3',
                spec=TensorSpec(dtype='float16', shape=(133, 14, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_3' in request_state_outputs,
            ),
        ),
        linear_3=_bind_tensor(
            linear_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_3',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_3' in request_state_outputs,
            ),
        ),
        reshape_4=_bind_tensor(
            reshape_4,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='reshape_4',
                spec=TensorSpec(dtype='float16', shape=(133, 14, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_4' in request_state_outputs,
            ),
        ),
        transpose_1=_bind_tensor(
            transpose_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='transpose_1',
                spec=TensorSpec(dtype='float16', shape=(14, 133, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='transpose_1' in request_state_outputs,
            ),
        ),
        unsqueeze=_bind_tensor(
            unsqueeze,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='unsqueeze',
                spec=TensorSpec(dtype='float16', shape=(1, 14, 133, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze' in request_state_outputs,
            ),
        ),
        transpose_2=_bind_tensor(
            transpose_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='transpose_2',
                spec=TensorSpec(dtype='float16', shape=(14, 133, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='transpose_2' in request_state_outputs,
            ),
        ),
        unsqueeze_1=_bind_tensor(
            unsqueeze_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='unsqueeze_1',
                spec=TensorSpec(dtype='float16', shape=(1, 14, 133, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze_1' in request_state_outputs,
            ),
        ),
        transpose_3=_bind_tensor(
            transpose_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='transpose_3',
                spec=TensorSpec(dtype='float16', shape=(14, 133, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='transpose_3' in request_state_outputs,
            ),
        ),
        unsqueeze_2=_bind_tensor(
            unsqueeze_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='unsqueeze_2',
                spec=TensorSpec(dtype='float16', shape=(1, 14, 133, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze_2' in request_state_outputs,
            ),
        ),
        scaled_dot_product_attention=_bind_tensor(
            scaled_dot_product_attention,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='scaled_dot_product_attention',
                spec=TensorSpec(dtype='float16', shape=(1, 14, 133, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='scaled_dot_product_attention' in request_state_outputs,
            ),
        ),
        transpose_4=_bind_tensor(
            transpose_4,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='transpose_4',
                spec=TensorSpec(dtype='float16', shape=(1, 133, 14, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='transpose_4' in request_state_outputs,
            ),
        ),
        contiguous=_bind_tensor(
            contiguous,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='contiguous',
                spec=TensorSpec(dtype='float16', shape=(1, 133, 14, 64)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='contiguous' in request_state_outputs,
            ),
        ),
        reshape_5=_bind_tensor(
            reshape_5,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='reshape_5',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_5' in request_state_outputs,
            ),
        ),
        linear_4=_bind_tensor(
            linear_4,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_4',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_4' in request_state_outputs,
            ),
        ),
        add_1=_bind_tensor(
            add_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add_1',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_1' in request_state_outputs,
            ),
        ),
        layer_norm_1=_bind_tensor(
            layer_norm_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='layer_norm_1',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_norm_1' in request_state_outputs,
            ),
        ),
        linear_5=_bind_tensor(
            linear_5,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_5',
                spec=TensorSpec(dtype='float16', shape=(133, 3584)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_5' in request_state_outputs,
            ),
        ),
        gelu_3=_bind_tensor(
            gelu_3,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='gelu_3',
                spec=TensorSpec(dtype='float16', shape=(133, 3584)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='gelu_3' in request_state_outputs,
            ),
        ),
        linear_6=_bind_tensor(
            linear_6,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_6',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_6' in request_state_outputs,
            ),
        ),
        add_2=_bind_tensor(
            add_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add_2',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_2' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    return tensors


@dataclass(frozen=True, slots=True)
class AudioEncoderTensors:
    p_audio_tower_ln_post_weight: LogicalTensor
    p_audio_tower_ln_post_bias: LogicalTensor
    p_audio_tower_conv2d1_weight: LogicalTensor
    p_audio_tower_conv2d1_bias: LogicalTensor
    p_audio_tower_conv2d2_weight: LogicalTensor
    p_audio_tower_conv2d2_bias: LogicalTensor
    p_audio_tower_conv2d3_weight: LogicalTensor
    p_audio_tower_conv2d3_bias: LogicalTensor
    p_audio_tower_conv_out_weight: LogicalTensor
    p_audio_tower_proj1_weight: LogicalTensor
    p_audio_tower_proj1_bias: LogicalTensor
    p_audio_tower_proj2_weight: LogicalTensor
    p_audio_tower_proj2_bias: LogicalTensor
    x: LogicalTensor
    position_embedding: LogicalTensor
    compact_index: LogicalTensor
    attention_mask: LogicalTensor
    conv2d: LogicalTensor
    gelu: LogicalTensor
    conv2d_1: LogicalTensor
    gelu_1: LogicalTensor
    conv2d_2: LogicalTensor
    gelu_2: LogicalTensor
    reshape: LogicalTensor
    transpose: LogicalTensor
    linear: LogicalTensor
    add: LogicalTensor
    reshape_1: LogicalTensor
    index_select: LogicalTensor
    layer_norm_36: LogicalTensor
    linear_109: LogicalTensor
    gelu_21: LogicalTensor
    linear_110: LogicalTensor
    layers: list[EncoderLayerTensors]


AUDIO_ENCODER_OUTPUT: str = 'linear_110'


def create_audio_encoder(
    prefix: str,
    *,
    p_audio_tower_ln_post_weight: LogicalTensor | None = None,
    p_audio_tower_ln_post_bias: LogicalTensor | None = None,
    p_audio_tower_conv2d1_weight: LogicalTensor | None = None,
    p_audio_tower_conv2d1_bias: LogicalTensor | None = None,
    p_audio_tower_conv2d2_weight: LogicalTensor | None = None,
    p_audio_tower_conv2d2_bias: LogicalTensor | None = None,
    p_audio_tower_conv2d3_weight: LogicalTensor | None = None,
    p_audio_tower_conv2d3_bias: LogicalTensor | None = None,
    p_audio_tower_conv_out_weight: LogicalTensor | None = None,
    p_audio_tower_proj1_weight: LogicalTensor | None = None,
    p_audio_tower_proj1_bias: LogicalTensor | None = None,
    p_audio_tower_proj2_weight: LogicalTensor | None = None,
    p_audio_tower_proj2_bias: LogicalTensor | None = None,
    x: LogicalTensor | None = None,
    position_embedding: LogicalTensor | None = None,
    compact_index: LogicalTensor | None = None,
    attention_mask: LogicalTensor | None = None,
    conv2d: LogicalTensor | None = None,
    gelu: LogicalTensor | None = None,
    conv2d_1: LogicalTensor | None = None,
    gelu_1: LogicalTensor | None = None,
    conv2d_2: LogicalTensor | None = None,
    gelu_2: LogicalTensor | None = None,
    reshape: LogicalTensor | None = None,
    transpose: LogicalTensor | None = None,
    linear: LogicalTensor | None = None,
    add: LogicalTensor | None = None,
    reshape_1: LogicalTensor | None = None,
    index_select: LogicalTensor | None = None,
    layer_norm_36: LogicalTensor | None = None,
    linear_109: LogicalTensor | None = None,
    gelu_21: LogicalTensor | None = None,
    linear_110: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> AudioEncoderTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('linear_110',)))
    tensors = AudioEncoderTensors(
        p_audio_tower_ln_post_weight=_bind_tensor(
            p_audio_tower_ln_post_weight,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.ln_post.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_ln_post_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_ln_post_bias=_bind_tensor(
            p_audio_tower_ln_post_bias,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.ln_post.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_ln_post_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_conv2d1_weight=_bind_tensor(
            p_audio_tower_conv2d1_weight,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.conv2d1.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(480, 1, 3, 3)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_conv2d1_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_conv2d1_bias=_bind_tensor(
            p_audio_tower_conv2d1_bias,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.conv2d1.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(480,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_conv2d1_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_conv2d2_weight=_bind_tensor(
            p_audio_tower_conv2d2_weight,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.conv2d2.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(480, 480, 3, 3)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_conv2d2_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_conv2d2_bias=_bind_tensor(
            p_audio_tower_conv2d2_bias,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.conv2d2.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(480,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_conv2d2_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_conv2d3_weight=_bind_tensor(
            p_audio_tower_conv2d3_weight,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.conv2d3.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(480, 480, 3, 3)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_conv2d3_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_conv2d3_bias=_bind_tensor(
            p_audio_tower_conv2d3_bias,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.conv2d3.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(480,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_conv2d3_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_conv_out_weight=_bind_tensor(
            p_audio_tower_conv_out_weight,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.conv_out.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896, 7680)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_conv_out_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_proj1_weight=_bind_tensor(
            p_audio_tower_proj1_weight,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.proj1.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_proj1_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_proj1_bias=_bind_tensor(
            p_audio_tower_proj1_bias,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.proj1.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(896,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_proj1_bias' in request_state_outputs,
            ),
        ),
        p_audio_tower_proj2_weight=_bind_tensor(
            p_audio_tower_proj2_weight,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.proj2.weight",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(1024, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_proj2_weight' in request_state_outputs,
            ),
        ),
        p_audio_tower_proj2_bias=_bind_tensor(
            p_audio_tower_proj2_bias,
            _declare_tensor(
                checkpoint_key="thinker.audio_tower.proj2.bias",
                reference_key=None,
                spec=TensorSpec(dtype='bfloat16', shape=(1024,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_audio_tower_proj2_bias' in request_state_outputs,
            ),
        ),
        x=_bind_tensor(
            x,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float16', shape=(11, 1, 128, 100)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='x' in request_state_outputs,
            ),
        ),
        position_embedding=_bind_tensor(
            position_embedding,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float16', shape=(11, 13, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='position_embedding' in request_state_outputs,
            ),
        ),
        compact_index=_bind_tensor(
            compact_index,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='int64', shape=(133,)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='compact_index' in request_state_outputs,
            ),
        ),
        attention_mask=_bind_tensor(
            attention_mask,
            _declare_tensor(
                checkpoint_key=None,
                reference_key=None,
                spec=TensorSpec(dtype='float16', shape=(1, 1, 133, 133)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='attention_mask' in request_state_outputs,
            ),
        ),
        conv2d=_bind_tensor(
            conv2d,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='conv2d',
                spec=TensorSpec(dtype='float16', shape=(11, 480, 64, 50)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d' in request_state_outputs,
            ),
        ),
        gelu=_bind_tensor(
            gelu,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='gelu',
                spec=TensorSpec(dtype='float16', shape=(11, 480, 64, 50)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='gelu' in request_state_outputs,
            ),
        ),
        conv2d_1=_bind_tensor(
            conv2d_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='conv2d_1',
                spec=TensorSpec(dtype='float16', shape=(11, 480, 32, 25)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_1' in request_state_outputs,
            ),
        ),
        gelu_1=_bind_tensor(
            gelu_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='gelu_1',
                spec=TensorSpec(dtype='float16', shape=(11, 480, 32, 25)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='gelu_1' in request_state_outputs,
            ),
        ),
        conv2d_2=_bind_tensor(
            conv2d_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='conv2d_2',
                spec=TensorSpec(dtype='float16', shape=(11, 480, 16, 13)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='conv2d_2' in request_state_outputs,
            ),
        ),
        gelu_2=_bind_tensor(
            gelu_2,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='gelu_2',
                spec=TensorSpec(dtype='float16', shape=(11, 480, 16, 13)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='gelu_2' in request_state_outputs,
            ),
        ),
        reshape=_bind_tensor(
            reshape,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='reshape',
                spec=TensorSpec(dtype='float16', shape=(11, 7680, 13)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape' in request_state_outputs,
            ),
        ),
        transpose=_bind_tensor(
            transpose,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='transpose',
                spec=TensorSpec(dtype='float16', shape=(11, 13, 7680)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='transpose' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            linear,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear',
                spec=TensorSpec(dtype='float16', shape=(11, 13, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear' in request_state_outputs,
            ),
        ),
        add=_bind_tensor(
            add,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='add',
                spec=TensorSpec(dtype='float16', shape=(11, 13, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add' in request_state_outputs,
            ),
        ),
        reshape_1=_bind_tensor(
            reshape_1,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='reshape_1',
                spec=TensorSpec(dtype='float16', shape=(143, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='reshape_1' in request_state_outputs,
            ),
        ),
        index_select=_bind_tensor(
            index_select,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='index_select',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='index_select' in request_state_outputs,
            ),
        ),
        layer_norm_36=_bind_tensor(
            layer_norm_36,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='layer_norm_36',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_norm_36' in request_state_outputs,
            ),
        ),
        linear_109=_bind_tensor(
            linear_109,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_109',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_109' in request_state_outputs,
            ),
        ),
        gelu_21=_bind_tensor(
            gelu_21,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='gelu_21',
                spec=TensorSpec(dtype='float16', shape=(133, 896)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='gelu_21' in request_state_outputs,
            ),
        ),
        linear_110=_bind_tensor(
            linear_110,
            _declare_tensor(
                checkpoint_key=None,
                reference_key='linear_110',
                spec=TensorSpec(dtype='float16', shape=(133, 1024)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_110' in request_state_outputs,
            ),
        ),
        layers=[create_encoder_layer(prefix, layer_idx=i) for i in range(18)],
    )
    bind_logical_tensor_names(tensors, prefix)
    _bind_alias_source(tensors.gelu_2, tensors.reshape)
    _bind_alias_source(tensors.add, tensors.reshape_1)
    _alias_carry = tensors.index_select
    for layer_t in tensors.layers:
        _bind_alias_source(layer_t.linear_1, layer_t.reshape_2)
        _bind_alias_source(layer_t.linear_2, layer_t.reshape_3)
        _bind_alias_source(layer_t.linear_3, layer_t.reshape_4)
        _bind_alias_source(layer_t.transpose_1, layer_t.unsqueeze)
        _bind_alias_source(layer_t.transpose_2, layer_t.unsqueeze_1)
        _bind_alias_source(layer_t.transpose_3, layer_t.unsqueeze_2)
        _bind_alias_source(layer_t.transpose_4, layer_t.contiguous)
        _bind_alias_source(layer_t.contiguous, layer_t.reshape_5)
        _alias_carry = layer_t.add_2
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
