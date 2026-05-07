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
class EncoderLayerTensors:
    p_attn_k_proj_weight: LogicalTensor
    p_attn_k_proj_bias: LogicalTensor
    p_attn_v_proj_weight: LogicalTensor
    p_attn_v_proj_bias: LogicalTensor
    p_attn_q_proj_weight: LogicalTensor
    p_attn_q_proj_bias: LogicalTensor
    p_attn_out_proj_weight: LogicalTensor
    p_attn_out_proj_bias: LogicalTensor
    p_attn_layer_norm_weight: LogicalTensor
    p_attn_layer_norm_bias: LogicalTensor
    p_fc1_weight: LogicalTensor
    p_fc1_bias: LogicalTensor
    p_fc2_weight: LogicalTensor
    p_fc2_bias: LogicalTensor
    p_final_layer_norm_weight: LogicalTensor
    p_final_layer_norm_bias: LogicalTensor
    hidden_states: LogicalTensor
    attention_mask: LogicalTensor
    layer_norm: LogicalTensor
    linear: LogicalTensor
    reshape: LogicalTensor
    linear_1: LogicalTensor
    reshape_1: LogicalTensor
    linear_2: LogicalTensor
    reshape_2: LogicalTensor
    transpose: LogicalTensor
    unsqueeze: LogicalTensor
    transpose_1: LogicalTensor
    unsqueeze_1: LogicalTensor
    transpose_2: LogicalTensor
    unsqueeze_2: LogicalTensor
    scaled_dot_product_attention: LogicalTensor
    transpose_3: LogicalTensor
    contiguous: LogicalTensor
    reshape_3: LogicalTensor
    linear_3: LogicalTensor
    add: LogicalTensor
    layer_norm_1: LogicalTensor
    linear_4: LogicalTensor
    gelu: LogicalTensor
    linear_5: LogicalTensor
    add_1: LogicalTensor


ENCODER_LAYER_OUTPUT: str = 'add_1'


def create_encoder_layer(prefix: str, layer_idx: int, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> EncoderLayerTensors:
    _validate_bindings(bindings, frozenset(('p_attn_k_proj_weight', 'p_attn_k_proj_bias', 'p_attn_v_proj_weight', 'p_attn_v_proj_bias', 'p_attn_q_proj_weight', 'p_attn_q_proj_bias', 'p_attn_out_proj_weight', 'p_attn_out_proj_bias', 'p_attn_layer_norm_weight', 'p_attn_layer_norm_bias', 'p_fc1_weight', 'p_fc1_bias', 'p_fc2_weight', 'p_fc2_bias', 'p_final_layer_norm_weight', 'p_final_layer_norm_bias', 'hidden_states', 'attention_mask', 'layer_norm', 'linear', 'reshape', 'linear_1', 'reshape_1', 'linear_2', 'reshape_2', 'transpose', 'unsqueeze', 'transpose_1', 'unsqueeze_1', 'transpose_2', 'unsqueeze_2', 'scaled_dot_product_attention', 'transpose_3', 'contiguous', 'reshape_3', 'linear_3', 'add', 'layer_norm_1', 'linear_4', 'gelu', 'linear_5', 'add_1')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('add_1',)))
    return EncoderLayerTensors(
        p_attn_k_proj_weight=_bind_tensor(
            bindings,
            'p_attn_k_proj_weight',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.self_attn.k_proj.weight",
            spec=TensorSpec(dtype='float32', shape=(896, 896)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_k_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_k_proj_bias=_bind_tensor(
            bindings,
            'p_attn_k_proj_bias',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.self_attn.k_proj.bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_k_proj_bias' in request_state_outputs,
            ),
        ),
        p_attn_v_proj_weight=_bind_tensor(
            bindings,
            'p_attn_v_proj_weight',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.self_attn.v_proj.weight",
            spec=TensorSpec(dtype='float32', shape=(896, 896)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_v_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_v_proj_bias=_bind_tensor(
            bindings,
            'p_attn_v_proj_bias',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.self_attn.v_proj.bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_v_proj_bias' in request_state_outputs,
            ),
        ),
        p_attn_q_proj_weight=_bind_tensor(
            bindings,
            'p_attn_q_proj_weight',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.self_attn.q_proj.weight",
            spec=TensorSpec(dtype='float32', shape=(896, 896)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_q_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_q_proj_bias=_bind_tensor(
            bindings,
            'p_attn_q_proj_bias',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.self_attn.q_proj.bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_q_proj_bias' in request_state_outputs,
            ),
        ),
        p_attn_out_proj_weight=_bind_tensor(
            bindings,
            'p_attn_out_proj_weight',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.self_attn.out_proj.weight",
            spec=TensorSpec(dtype='float32', shape=(896, 896)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_out_proj_weight' in request_state_outputs,
            ),
        ),
        p_attn_out_proj_bias=_bind_tensor(
            bindings,
            'p_attn_out_proj_bias',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.self_attn.out_proj.bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_out_proj_bias' in request_state_outputs,
            ),
        ),
        p_attn_layer_norm_weight=_bind_tensor(
            bindings,
            'p_attn_layer_norm_weight',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.self_attn_layer_norm.weight",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_layer_norm_weight' in request_state_outputs,
            ),
        ),
        p_attn_layer_norm_bias=_bind_tensor(
            bindings,
            'p_attn_layer_norm_bias',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.self_attn_layer_norm.bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_attn_layer_norm_bias' in request_state_outputs,
            ),
        ),
        p_fc1_weight=_bind_tensor(
            bindings,
            'p_fc1_weight',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.fc1.weight",
            spec=TensorSpec(dtype='float32', shape=(3584, 896)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_fc1_weight' in request_state_outputs,
            ),
        ),
        p_fc1_bias=_bind_tensor(
            bindings,
            'p_fc1_bias',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.fc1.bias",
            spec=TensorSpec(dtype='float32', shape=(3584,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_fc1_bias' in request_state_outputs,
            ),
        ),
        p_fc2_weight=_bind_tensor(
            bindings,
            'p_fc2_weight',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.fc2.weight",
            spec=TensorSpec(dtype='float32', shape=(896, 3584)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_fc2_weight' in request_state_outputs,
            ),
        ),
        p_fc2_bias=_bind_tensor(
            bindings,
            'p_fc2_bias',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.fc2.bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_fc2_bias' in request_state_outputs,
            ),
        ),
        p_final_layer_norm_weight=_bind_tensor(
            bindings,
            'p_final_layer_norm_weight',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.final_layer_norm.weight",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_final_layer_norm_weight' in request_state_outputs,
            ),
        ),
        p_final_layer_norm_bias=_bind_tensor(
            bindings,
            'p_final_layer_norm_bias',
            _declare_tensor(
            name=f"thinker.audio_tower.layers.{layer_idx}.final_layer_norm.bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_final_layer_norm_bias' in request_state_outputs,
            ),
        ),
        hidden_states=_bind_tensor(
            bindings,
            'hidden_states',
            _declare_tensor(
            name=f"{prefix}.hidden_states",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='hidden_states' in request_state_outputs,
            ),
        ),
        attention_mask=_bind_tensor(
            bindings,
            'attention_mask',
            _declare_tensor(
            name=f"{prefix}.attention_mask",
            spec=TensorSpec(dtype='float32', shape=(1, 1, 133, 133)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='attention_mask' in request_state_outputs,
            ),
        ),
        layer_norm=_bind_tensor(
            bindings,
            'layer_norm',
            _declare_tensor(
            name=f"{prefix}.layer_norm",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='layer_norm' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            bindings,
            'linear',
            _declare_tensor(
            name=f"{prefix}.linear",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear' in request_state_outputs,
            ),
        ),
        reshape=_bind_tensor(
            bindings,
            'reshape',
            _declare_tensor(
            name=f"{prefix}.reshape",
            spec=TensorSpec(dtype='float32', shape=(133, 14, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='reshape' in request_state_outputs,
            ),
        ),
        linear_1=_bind_tensor(
            bindings,
            'linear_1',
            _declare_tensor(
            name=f"{prefix}.linear_1",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_1' in request_state_outputs,
            ),
        ),
        reshape_1=_bind_tensor(
            bindings,
            'reshape_1',
            _declare_tensor(
            name=f"{prefix}.reshape_1",
            spec=TensorSpec(dtype='float32', shape=(133, 14, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='reshape_1' in request_state_outputs,
            ),
        ),
        linear_2=_bind_tensor(
            bindings,
            'linear_2',
            _declare_tensor(
            name=f"{prefix}.linear_2",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_2' in request_state_outputs,
            ),
        ),
        reshape_2=_bind_tensor(
            bindings,
            'reshape_2',
            _declare_tensor(
            name=f"{prefix}.reshape_2",
            spec=TensorSpec(dtype='float32', shape=(133, 14, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='reshape_2' in request_state_outputs,
            ),
        ),
        transpose=_bind_tensor(
            bindings,
            'transpose',
            _declare_tensor(
            name=f"{prefix}.transpose",
            spec=TensorSpec(dtype='float32', shape=(14, 133, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='transpose' in request_state_outputs,
            ),
        ),
        unsqueeze=_bind_tensor(
            bindings,
            'unsqueeze',
            _declare_tensor(
            name=f"{prefix}.unsqueeze",
            spec=TensorSpec(dtype='float32', shape=(1, 14, 133, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='unsqueeze' in request_state_outputs,
            ),
        ),
        transpose_1=_bind_tensor(
            bindings,
            'transpose_1',
            _declare_tensor(
            name=f"{prefix}.transpose_1",
            spec=TensorSpec(dtype='float32', shape=(14, 133, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='transpose_1' in request_state_outputs,
            ),
        ),
        unsqueeze_1=_bind_tensor(
            bindings,
            'unsqueeze_1',
            _declare_tensor(
            name=f"{prefix}.unsqueeze_1",
            spec=TensorSpec(dtype='float32', shape=(1, 14, 133, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='unsqueeze_1' in request_state_outputs,
            ),
        ),
        transpose_2=_bind_tensor(
            bindings,
            'transpose_2',
            _declare_tensor(
            name=f"{prefix}.transpose_2",
            spec=TensorSpec(dtype='float32', shape=(14, 133, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='transpose_2' in request_state_outputs,
            ),
        ),
        unsqueeze_2=_bind_tensor(
            bindings,
            'unsqueeze_2',
            _declare_tensor(
            name=f"{prefix}.unsqueeze_2",
            spec=TensorSpec(dtype='float32', shape=(1, 14, 133, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='unsqueeze_2' in request_state_outputs,
            ),
        ),
        scaled_dot_product_attention=_bind_tensor(
            bindings,
            'scaled_dot_product_attention',
            _declare_tensor(
            name=f"{prefix}.scaled_dot_product_attention",
            spec=TensorSpec(dtype='float32', shape=(1, 14, 133, 64)),
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
            spec=TensorSpec(dtype='float32', shape=(1, 133, 14, 64)),
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
            spec=TensorSpec(dtype='float32', shape=(1, 133, 14, 64)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='contiguous' in request_state_outputs,
            ),
        ),
        reshape_3=_bind_tensor(
            bindings,
            'reshape_3',
            _declare_tensor(
            name=f"{prefix}.reshape_3",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='reshape_3' in request_state_outputs,
            ),
        ),
        linear_3=_bind_tensor(
            bindings,
            'linear_3',
            _declare_tensor(
            name=f"{prefix}.linear_3",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_3' in request_state_outputs,
            ),
        ),
        add=_bind_tensor(
            bindings,
            'add',
            _declare_tensor(
            name=f"{prefix}.add",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add' in request_state_outputs,
            ),
        ),
        layer_norm_1=_bind_tensor(
            bindings,
            'layer_norm_1',
            _declare_tensor(
            name=f"{prefix}.layer_norm_1",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='layer_norm_1' in request_state_outputs,
            ),
        ),
        linear_4=_bind_tensor(
            bindings,
            'linear_4',
            _declare_tensor(
            name=f"{prefix}.linear_4",
            spec=TensorSpec(dtype='float32', shape=(133, 3584)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_4' in request_state_outputs,
            ),
        ),
        gelu=_bind_tensor(
            bindings,
            'gelu',
            _declare_tensor(
            name=f"{prefix}.gelu",
            spec=TensorSpec(dtype='float32', shape=(133, 3584)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='gelu' in request_state_outputs,
            ),
        ),
        linear_5=_bind_tensor(
            bindings,
            'linear_5',
            _declare_tensor(
            name=f"{prefix}.linear_5",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_5' in request_state_outputs,
            ),
        ),
        add_1=_bind_tensor(
            bindings,
            'add_1',
            _declare_tensor(
            name=f"{prefix}.add_1",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add_1' in request_state_outputs,
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
