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
class Conv2d1Tensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    x: LogicalTensor
    conv2d: LogicalTensor
    gelu: LogicalTensor


CONV2D1_OUTPUT: str = 'gelu'


def create_conv2d1(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> Conv2d1Tensors:
    _validate_bindings(bindings, frozenset(('p_weight', 'p_bias', 'x', 'conv2d', 'gelu')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('gelu',)))
    return Conv2d1Tensors(
        p_weight=_bind_tensor(
            bindings,
            'p_weight',
            _declare_tensor(
            name="thinker.audio_tower.conv2d1.weight",
            spec=TensorSpec(dtype='float32', shape=(480, 1, 3, 3)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_weight' in request_state_outputs,
            ),
        ),
        p_bias=_bind_tensor(
            bindings,
            'p_bias',
            _declare_tensor(
            name="thinker.audio_tower.conv2d1.bias",
            spec=TensorSpec(dtype='float32', shape=(480,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_bias' in request_state_outputs,
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
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class Conv2d2Tensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    x: LogicalTensor
    conv2d: LogicalTensor
    gelu: LogicalTensor


CONV2D2_OUTPUT: str = 'gelu'


def create_conv2d2(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> Conv2d2Tensors:
    _validate_bindings(bindings, frozenset(('p_weight', 'p_bias', 'x', 'conv2d', 'gelu')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('gelu',)))
    return Conv2d2Tensors(
        p_weight=_bind_tensor(
            bindings,
            'p_weight',
            _declare_tensor(
            name="thinker.audio_tower.conv2d2.weight",
            spec=TensorSpec(dtype='float32', shape=(480, 480, 3, 3)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_weight' in request_state_outputs,
            ),
        ),
        p_bias=_bind_tensor(
            bindings,
            'p_bias',
            _declare_tensor(
            name="thinker.audio_tower.conv2d2.bias",
            spec=TensorSpec(dtype='float32', shape=(480,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_bias' in request_state_outputs,
            ),
        ),
        x=_bind_tensor(
            bindings,
            'x',
            _declare_tensor(
            name=f"{prefix}.x",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 64, 50)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='x' in request_state_outputs,
            ),
        ),
        conv2d=_bind_tensor(
            bindings,
            'conv2d',
            _declare_tensor(
            name=f"{prefix}.conv2d",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 32, 25)),
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
            spec=TensorSpec(dtype='float32', shape=(11, 480, 32, 25)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='gelu' in request_state_outputs,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class Conv2d3Tensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    x: LogicalTensor
    conv2d: LogicalTensor
    gelu: LogicalTensor


CONV2D3_OUTPUT: str = 'gelu'


def create_conv2d3(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> Conv2d3Tensors:
    _validate_bindings(bindings, frozenset(('p_weight', 'p_bias', 'x', 'conv2d', 'gelu')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('gelu',)))
    return Conv2d3Tensors(
        p_weight=_bind_tensor(
            bindings,
            'p_weight',
            _declare_tensor(
            name="thinker.audio_tower.conv2d3.weight",
            spec=TensorSpec(dtype='float32', shape=(480, 480, 3, 3)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_weight' in request_state_outputs,
            ),
        ),
        p_bias=_bind_tensor(
            bindings,
            'p_bias',
            _declare_tensor(
            name="thinker.audio_tower.conv2d3.bias",
            spec=TensorSpec(dtype='float32', shape=(480,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_bias' in request_state_outputs,
            ),
        ),
        x=_bind_tensor(
            bindings,
            'x',
            _declare_tensor(
            name=f"{prefix}.x",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 32, 25)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='x' in request_state_outputs,
            ),
        ),
        conv2d=_bind_tensor(
            bindings,
            'conv2d',
            _declare_tensor(
            name=f"{prefix}.conv2d",
            spec=TensorSpec(dtype='float32', shape=(11, 480, 16, 13)),
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
            spec=TensorSpec(dtype='float32', shape=(11, 480, 16, 13)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='gelu' in request_state_outputs,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class ConvOutTensors:
    p_weight: LogicalTensor
    input: LogicalTensor
    linear: LogicalTensor


CONV_OUT_OUTPUT: str = 'linear'


def create_conv_out(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> ConvOutTensors:
    _validate_bindings(bindings, frozenset(('p_weight', 'input', 'linear')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('linear',)))
    return ConvOutTensors(
        p_weight=_bind_tensor(
            bindings,
            'p_weight',
            _declare_tensor(
            name="thinker.audio_tower.conv_out.weight",
            spec=TensorSpec(dtype='float32', shape=(896, 7680)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_weight' in request_state_outputs,
            ),
        ),
        input=_bind_tensor(
            bindings,
            'input',
            _declare_tensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='float32', shape=(143, 7680)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='input' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            bindings,
            'linear',
            _declare_tensor(
            name=f"{prefix}.linear",
            spec=TensorSpec(dtype='float32', shape=(143, 896)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear' in request_state_outputs,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class LnPostTensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    input: LogicalTensor
    layer_norm: LogicalTensor


LN_POST_OUTPUT: str = 'layer_norm'


def create_ln_post(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> LnPostTensors:
    _validate_bindings(bindings, frozenset(('p_weight', 'p_bias', 'input', 'layer_norm')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('layer_norm',)))
    return LnPostTensors(
        p_weight=_bind_tensor(
            bindings,
            'p_weight',
            _declare_tensor(
            name="thinker.audio_tower.ln_post.weight",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_weight' in request_state_outputs,
            ),
        ),
        p_bias=_bind_tensor(
            bindings,
            'p_bias',
            _declare_tensor(
            name="thinker.audio_tower.ln_post.bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_bias' in request_state_outputs,
            ),
        ),
        input=_bind_tensor(
            bindings,
            'input',
            _declare_tensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='input' in request_state_outputs,
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
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class Proj1Tensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    x: LogicalTensor
    linear: LogicalTensor
    gelu: LogicalTensor


PROJ1_OUTPUT: str = 'gelu'


def create_proj1(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> Proj1Tensors:
    _validate_bindings(bindings, frozenset(('p_weight', 'p_bias', 'x', 'linear', 'gelu')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('gelu',)))
    return Proj1Tensors(
        p_weight=_bind_tensor(
            bindings,
            'p_weight',
            _declare_tensor(
            name="thinker.audio_tower.proj1.weight",
            spec=TensorSpec(dtype='float32', shape=(896, 896)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_weight' in request_state_outputs,
            ),
        ),
        p_bias=_bind_tensor(
            bindings,
            'p_bias',
            _declare_tensor(
            name="thinker.audio_tower.proj1.bias",
            spec=TensorSpec(dtype='float32', shape=(896,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_bias' in request_state_outputs,
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
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class Proj2Tensors:
    p_weight: LogicalTensor
    p_bias: LogicalTensor
    input: LogicalTensor
    linear: LogicalTensor


PROJ2_OUTPUT: str = 'linear'


def create_proj2(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> Proj2Tensors:
    _validate_bindings(bindings, frozenset(('p_weight', 'p_bias', 'input', 'linear')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('linear',)))
    return Proj2Tensors(
        p_weight=_bind_tensor(
            bindings,
            'p_weight',
            _declare_tensor(
            name="thinker.audio_tower.proj2.weight",
            spec=TensorSpec(dtype='float32', shape=(1024, 896)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_weight' in request_state_outputs,
            ),
        ),
        p_bias=_bind_tensor(
            bindings,
            'p_bias',
            _declare_tensor(
            name="thinker.audio_tower.proj2.bias",
            spec=TensorSpec(dtype='float32', shape=(1024,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_bias' in request_state_outputs,
            ),
        ),
        input=_bind_tensor(
            bindings,
            'input',
            _declare_tensor(
            name=f"{prefix}.input",
            spec=TensorSpec(dtype='float32', shape=(133, 896)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='input' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            bindings,
            'linear',
            _declare_tensor(
            name=f"{prefix}.linear",
            spec=TensorSpec(dtype='float32', shape=(133, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='linear' in request_state_outputs,
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
