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
class AEEntryTensors:
    tokens: LogicalTensor
    to: LogicalTensor
    permute: LogicalTensor
    contiguous: LogicalTensor


AE_ENTRY_OUTPUT: str = 'contiguous'


def create_ae_entry(
    prefix: str,
    *,
    latent_height: int,
    latent_width: int,
    tokens: LogicalTensor | None = None,
    to: LogicalTensor | None = None,
    permute: LogicalTensor | None = None,
    contiguous: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> AEEntryTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('contiguous',)))
    tensors = AEEntryTensors(
        tokens=_bind_tensor(
            tokens,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=prefix,
                spec=TensorSpec(dtype='float32', shape=(1, latent_height, latent_width, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='tokens' in request_state_outputs,
            ),
        ),
        to=_bind_tensor(
            to,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='to',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, latent_height, latent_width, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='to' in request_state_outputs,
            ),
        ),
        permute=_bind_tensor(
            permute,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='permute',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 128, latent_height, latent_width)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='permute' in request_state_outputs,
            ),
        ),
        contiguous=_bind_tensor(
            contiguous,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='contiguous',
                layer=prefix,
                spec=TensorSpec(dtype='float16', shape=(1, 128, latent_height, latent_width)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='contiguous' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    _bind_alias_source(tensors.permute, tensors.contiguous)
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
