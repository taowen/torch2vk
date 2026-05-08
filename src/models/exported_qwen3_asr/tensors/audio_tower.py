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
class AudioConvStackTensors:
    p_conv2d1_weight: LogicalTensor
    p_conv2d1_bias: LogicalTensor
    p_conv2d2_weight: LogicalTensor
    p_conv2d2_bias: LogicalTensor
    p_conv2d3_weight: LogicalTensor
    p_conv2d3_bias: LogicalTensor
    p_conv_out_weight: LogicalTensor
    x: LogicalTensor
    position_embedding: LogicalTensor
    compact_index: LogicalTensor
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


AUDIO_CONV_STACK_OUTPUT: str = 'index_select'


def create_audio_conv_stack(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> AudioConvStackTensors:
    _validate_bindings(bindings, frozenset(('p_conv2d1_weight', 'p_conv2d1_bias', 'p_conv2d2_weight', 'p_conv2d2_bias', 'p_conv2d3_weight', 'p_conv2d3_bias', 'p_conv_out_weight', 'x', 'position_embedding', 'compact_index', 'conv2d', 'gelu', 'conv2d_1', 'gelu_1', 'conv2d_2', 'gelu_2', 'reshape', 'transpose', 'linear', 'add', 'reshape_1', 'index_select')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('index_select',)))
    return AudioConvStackTensors(
        p_conv2d1_weight=_bind_tensor(
            bindings,
            'p_conv2d1_weight',
            _declare_tensor(
            name="thinker.audio_tower.conv2d1.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(480, 1, 3, 3)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_conv2d1_weight' in request_state_outputs,
            ),
        ),
        p_conv2d1_bias=_bind_tensor(
            bindings,
            'p_conv2d1_bias',
            _declare_tensor(
            name="thinker.audio_tower.conv2d1.bias",
            spec=TensorSpec(dtype='bfloat16', shape=(480,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_conv2d1_bias' in request_state_outputs,
            ),
        ),
        p_conv2d2_weight=_bind_tensor(
            bindings,
            'p_conv2d2_weight',
            _declare_tensor(
            name="thinker.audio_tower.conv2d2.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(480, 480, 3, 3)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_conv2d2_weight' in request_state_outputs,
            ),
        ),
        p_conv2d2_bias=_bind_tensor(
            bindings,
            'p_conv2d2_bias',
            _declare_tensor(
            name="thinker.audio_tower.conv2d2.bias",
            spec=TensorSpec(dtype='bfloat16', shape=(480,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_conv2d2_bias' in request_state_outputs,
            ),
        ),
        p_conv2d3_weight=_bind_tensor(
            bindings,
            'p_conv2d3_weight',
            _declare_tensor(
            name="thinker.audio_tower.conv2d3.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(480, 480, 3, 3)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_conv2d3_weight' in request_state_outputs,
            ),
        ),
        p_conv2d3_bias=_bind_tensor(
            bindings,
            'p_conv2d3_bias',
            _declare_tensor(
            name="thinker.audio_tower.conv2d3.bias",
            spec=TensorSpec(dtype='bfloat16', shape=(480,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_conv2d3_bias' in request_state_outputs,
            ),
        ),
        p_conv_out_weight=_bind_tensor(
            bindings,
            'p_conv_out_weight',
            _declare_tensor(
            name="thinker.audio_tower.conv_out.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(896, 7680)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_conv_out_weight' in request_state_outputs,
            ),
        ),
        x=_bind_tensor(
            bindings,
            'x',
            _declare_tensor(
            name=f"{prefix}.x",
            spec=TensorSpec(dtype='float32', shape=(11, 1, 128, 100)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='x' in request_state_outputs,
            ),
        ),
        position_embedding=_bind_tensor(
            bindings,
            'position_embedding',
            _declare_tensor(
            name=f"{prefix}.position_embedding",
            spec=TensorSpec(dtype='float32', shape=(11, 13, 896)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='position_embedding' in request_state_outputs,
            ),
        ),
        compact_index=_bind_tensor(
            bindings,
            'compact_index',
            _declare_tensor(
            name=f"{prefix}.compact_index",
            spec=TensorSpec(dtype='int64', shape=(133,)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='compact_index' in request_state_outputs,
            ),
        ),
        conv2d=_bind_tensor(
            bindings,
            'conv2d',
            _declare_tensor(
            name=f"{prefix}.conv2d",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 64, 50)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='conv2d' in request_state_outputs,
            ),
        ),
        gelu=_bind_tensor(
            bindings,
            'gelu',
            _declare_tensor(
            name=f"{prefix}.gelu",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 64, 50)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='gelu' in request_state_outputs,
            ),
        ),
        conv2d_1=_bind_tensor(
            bindings,
            'conv2d_1',
            _declare_tensor(
            name=f"{prefix}.conv2d_1",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 32, 25)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='conv2d_1' in request_state_outputs,
            ),
        ),
        gelu_1=_bind_tensor(
            bindings,
            'gelu_1',
            _declare_tensor(
            name=f"{prefix}.gelu_1",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 32, 25)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='gelu_1' in request_state_outputs,
            ),
        ),
        conv2d_2=_bind_tensor(
            bindings,
            'conv2d_2',
            _declare_tensor(
            name=f"{prefix}.conv2d_2",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 16, 13)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='conv2d_2' in request_state_outputs,
            ),
        ),
        gelu_2=_bind_tensor(
            bindings,
            'gelu_2',
            _declare_tensor(
            name=f"{prefix}.gelu_2",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 16, 13)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='gelu_2' in request_state_outputs,
            ),
        ),
        reshape=_bind_tensor(
            bindings,
            'reshape',
            _declare_tensor(
            name=f"{prefix}.reshape",
            spec=TensorSpec(dtype='float32', shape=(11, 7680, 13)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='reshape' in request_state_outputs,
            ),
        ),
        transpose=_bind_tensor(
            bindings,
            'transpose',
            _declare_tensor(
            name=f"{prefix}.transpose",
            spec=TensorSpec(dtype='float32', shape=(11, 13, 7680)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='transpose' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            bindings,
            'linear',
            _declare_tensor(
            name=f"{prefix}.linear",
            spec=TensorSpec(dtype='float32', shape=(11, 13, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear' in request_state_outputs,
            ),
        ),
        add=_bind_tensor(
            bindings,
            'add',
            _declare_tensor(
            name=f"{prefix}.add",
            spec=TensorSpec(dtype='float32', shape=(11, 13, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='add' in request_state_outputs,
            ),
        ),
        reshape_1=_bind_tensor(
            bindings,
            'reshape_1',
            _declare_tensor(
            name=f"{prefix}.reshape_1",
            spec=TensorSpec(dtype='float32', shape=(143, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='reshape_1' in request_state_outputs,
            ),
        ),
        index_select=_bind_tensor(
            bindings,
            'index_select',
            _declare_tensor(
            name=f"{prefix}.index_select",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='index_select' in request_state_outputs,
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class AudioProjTensors:
    p_ln_post_weight: LogicalTensor
    p_ln_post_bias: LogicalTensor
    p_proj1_weight: LogicalTensor
    p_proj1_bias: LogicalTensor
    p_proj2_weight: LogicalTensor
    p_proj2_bias: LogicalTensor
    x: LogicalTensor
    layer_norm: LogicalTensor
    linear: LogicalTensor
    gelu: LogicalTensor
    linear_1: LogicalTensor


AUDIO_PROJ_OUTPUT: str = 'linear_1'


def create_audio_proj(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> AudioProjTensors:
    _validate_bindings(bindings, frozenset(('p_ln_post_weight', 'p_ln_post_bias', 'p_proj1_weight', 'p_proj1_bias', 'p_proj2_weight', 'p_proj2_bias', 'x', 'layer_norm', 'linear', 'gelu', 'linear_1')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('linear_1',)))
    return AudioProjTensors(
        p_ln_post_weight=_bind_tensor(
            bindings,
            'p_ln_post_weight',
            _declare_tensor(
            name="thinker.audio_tower.ln_post.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_ln_post_weight' in request_state_outputs,
            ),
        ),
        p_ln_post_bias=_bind_tensor(
            bindings,
            'p_ln_post_bias',
            _declare_tensor(
            name="thinker.audio_tower.ln_post.bias",
            spec=TensorSpec(dtype='bfloat16', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_ln_post_bias' in request_state_outputs,
            ),
        ),
        p_proj1_weight=_bind_tensor(
            bindings,
            'p_proj1_weight',
            _declare_tensor(
            name="thinker.audio_tower.proj1.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(896, 896)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_proj1_weight' in request_state_outputs,
            ),
        ),
        p_proj1_bias=_bind_tensor(
            bindings,
            'p_proj1_bias',
            _declare_tensor(
            name="thinker.audio_tower.proj1.bias",
            spec=TensorSpec(dtype='bfloat16', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_proj1_bias' in request_state_outputs,
            ),
        ),
        p_proj2_weight=_bind_tensor(
            bindings,
            'p_proj2_weight',
            _declare_tensor(
            name="thinker.audio_tower.proj2.weight",
            spec=TensorSpec(dtype='bfloat16', shape=(1024, 896)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_proj2_weight' in request_state_outputs,
            ),
        ),
        p_proj2_bias=_bind_tensor(
            bindings,
            'p_proj2_bias',
            _declare_tensor(
            name="thinker.audio_tower.proj2.bias",
            spec=TensorSpec(dtype='bfloat16', shape=(1024,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_proj2_bias' in request_state_outputs,
            ),
        ),
        x=_bind_tensor(
            bindings,
            'x',
            _declare_tensor(
            name=f"{prefix}.x",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='x' in request_state_outputs,
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
        gelu=_bind_tensor(
            bindings,
            'gelu',
            _declare_tensor(
            name=f"{prefix}.gelu",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='gelu' in request_state_outputs,
            ),
        ),
        linear_1=_bind_tensor(
            bindings,
            'linear_1',
            _declare_tensor(
            name=f"{prefix}.linear_1",
            spec=TensorSpec(dtype='float32', shape=(133, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear_1' in request_state_outputs,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", selector="last_hidden_state"),
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
