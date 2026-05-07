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
class EmbedTokensTensors:
    p_weight: LogicalTensor
    input: LogicalTensor
    embedding: LogicalTensor


EMBED_TOKENS_OUTPUT: str = 'embedding'


def create_embed_tokens(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> EmbedTokensTensors:
    _validate_bindings(bindings, frozenset(('p_weight', 'input', 'embedding')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('embedding',)))
    return EmbedTokensTensors(
        p_weight=_bind_tensor(
            bindings,
            'p_weight',
            _declare_tensor(
            name="thinker.model.embed_tokens.weight",
            spec=TensorSpec(dtype='float32', shape=(151936, 1024)),
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
            spec=TensorSpec(dtype='int32', shape=(1, 151)),
            role=TensorRole.INPUT,
            memory=MemoryClass.HOST_INPUT,
            lifetime=TensorLifetime.FRAME,
            request_state='input' in request_state_outputs,
            ),
        ),
        embedding=_bind_tensor(
            bindings,
            'embedding',
            _declare_tensor(
            name=f"{prefix}.embedding",
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
            role=TensorRole.ACTIVATION,
            memory=MemoryClass.FRAME_WORKSPACE,
            lifetime=TensorLifetime.FRAME,
            request_state='embedding' in request_state_outputs,
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class TextNormTensors:
    p_weight: LogicalTensor
    hidden_states: LogicalTensor
    to: LogicalTensor
    pow_1: LogicalTensor
    mean: LogicalTensor
    add: LogicalTensor
    rsqrt: LogicalTensor
    mul: LogicalTensor
    to_1: LogicalTensor
    mul_1: LogicalTensor


TEXT_NORM_OUTPUT: str = 'mul_1'


def create_text_norm(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> TextNormTensors:
    _validate_bindings(bindings, frozenset(('p_weight', 'hidden_states', 'to', 'pow_1', 'mean', 'add', 'rsqrt', 'mul', 'to_1', 'mul_1')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('mul_1',)))
    return TextNormTensors(
        p_weight=_bind_tensor(
            bindings,
            'p_weight',
            _declare_tensor(
            name="thinker.model.norm.weight",
            spec=TensorSpec(dtype='float32', shape=(1024,)),
            role=TensorRole.WEIGHT,
            memory=MemoryClass.MODEL_WEIGHT,
            lifetime=TensorLifetime.MODEL,
            request_state='p_weight' in request_state_outputs,
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
            compare=ComparePolicy(kind="tensor", rtol=3e-3, atol=3e-2),
            pytorch_probe=PyTorchProbe(kind="module_output", target="", index=0),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class LmHeadTensors:
    p_weight: LogicalTensor
    input: LogicalTensor
    linear: LogicalTensor


LM_HEAD_OUTPUT: str = 'linear'


def create_lm_head(prefix: str, *, bindings: Mapping[str, LogicalTensor] | None = None, request_state_outputs: Collection[str] = frozenset()) -> LmHeadTensors:
    _validate_bindings(bindings, frozenset(('p_weight', 'input', 'linear')))
    _validate_request_state_outputs(request_state_outputs, frozenset(('linear',)))
    return LmHeadTensors(
        p_weight=_bind_tensor(
            bindings,
            'p_weight',
            _declare_tensor(
            name="thinker.lm_head.weight",
            spec=TensorSpec(dtype='float32', shape=(151936, 1024)),
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
            spec=TensorSpec(dtype='float32', shape=(1, 151, 1024)),
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
            spec=TensorSpec(dtype='float32', shape=(1, 151, 151936)),
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
