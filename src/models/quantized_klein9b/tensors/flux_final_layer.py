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
class FluxFinalLayerTensors:
    p_final_layer_linear_weight: LogicalTensor
    p_final_layer_adaln_modulation_1_weight: LogicalTensor
    hidden_states: LogicalTensor
    vec: LogicalTensor
    slice_1: LogicalTensor
    silu: LogicalTensor
    linear: LogicalTensor
    getitem: LogicalTensor
    getitem_1: LogicalTensor
    unsqueeze: LogicalTensor
    unsqueeze_1: LogicalTensor
    add: LogicalTensor
    layer_norm: LogicalTensor
    mul: LogicalTensor
    add_1: LogicalTensor
    linear_1: LogicalTensor


FLUX_FINAL_LAYER_OUTPUT: str = 'linear_1'


def create_flux_final_layer(
    prefix: str,
    *,
    text_seq_len: int,
    image_seq_len: int,
    p_final_layer_linear_weight: LogicalTensor | None = None,
    p_final_layer_adaln_modulation_1_weight: LogicalTensor | None = None,
    hidden_states: LogicalTensor | None = None,
    vec: LogicalTensor | None = None,
    slice_1: LogicalTensor | None = None,
    silu: LogicalTensor | None = None,
    linear: LogicalTensor | None = None,
    getitem: LogicalTensor | None = None,
    getitem_1: LogicalTensor | None = None,
    unsqueeze: LogicalTensor | None = None,
    unsqueeze_1: LogicalTensor | None = None,
    add: LogicalTensor | None = None,
    layer_norm: LogicalTensor | None = None,
    mul: LogicalTensor | None = None,
    add_1: LogicalTensor | None = None,
    linear_1: LogicalTensor | None = None,
    request_state_outputs: Collection[str] = frozenset(),
) -> FluxFinalLayerTensors:
    _validate_request_state_outputs(request_state_outputs, frozenset(('linear_1',)))
    tensors = FluxFinalLayerTensors(
        p_final_layer_linear_weight=_bind_tensor(
            p_final_layer_linear_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key="final_layer.linear.weight",
                reference_key=None,
                layer=None,
                spec=_quantized_weight_spec("final_layer.linear.weight", dtype='float16', shape=(128, 4096)),
                layout=_quantized_weight_layout("final_layer.linear.weight", dtype='float16', shape=(128, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_final_layer_linear_weight' in request_state_outputs,
            ),
        ),
        p_final_layer_adaln_modulation_1_weight=_bind_tensor(
            p_final_layer_adaln_modulation_1_weight,
            _declare_tensor(
                checkpoint='flux/model.gguf',
                checkpoint_key="final_layer.adaLN_modulation.1.weight",
                reference_key=None,
                layer=None,
                spec=_quantized_weight_spec("final_layer.adaLN_modulation.1.weight", dtype='float16', shape=(8192, 4096)),
                layout=_quantized_weight_layout("final_layer.adaLN_modulation.1.weight", dtype='float16', shape=(8192, 4096)),
                role=TensorRole.WEIGHT,
                memory=MemoryClass.MODEL_WEIGHT,
                lifetime=TensorLifetime.MODEL,
                request_state='p_final_layer_adaln_modulation_1_weight' in request_state_outputs,
            ),
        ),
        hidden_states=_bind_tensor(
            hidden_states,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, text_seq_len + image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='hidden_states' in request_state_outputs,
            ),
        ),
        vec=_bind_tensor(
            vec,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key=None,
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.INPUT,
                memory=MemoryClass.HOST_INPUT,
                lifetime=TensorLifetime.FRAME,
                request_state='vec' in request_state_outputs,
            ),
        ),
        slice_1=_bind_tensor(
            slice_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='slice_1',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='slice_1' in request_state_outputs,
            ),
        ),
        silu=_bind_tensor(
            silu,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='silu',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='silu' in request_state_outputs,
            ),
        ),
        linear=_bind_tensor(
            linear,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, 8192)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear' in request_state_outputs,
            ),
        ),
        getitem=_bind_tensor(
            getitem,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem' in request_state_outputs,
            ),
        ),
        getitem_1=_bind_tensor(
            getitem_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='getitem_1',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='getitem_1' in request_state_outputs,
            ),
        ),
        unsqueeze=_bind_tensor(
            unsqueeze,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='unsqueeze',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
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
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='unsqueeze_1' in request_state_outputs,
            ),
        ),
        add=_bind_tensor(
            add,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, 1, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add' in request_state_outputs,
            ),
        ),
        layer_norm=_bind_tensor(
            layer_norm,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='layer_norm',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='layer_norm' in request_state_outputs,
            ),
        ),
        mul=_bind_tensor(
            mul,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='mul',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='mul' in request_state_outputs,
            ),
        ),
        add_1=_bind_tensor(
            add_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='add_1',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 4096)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='add_1' in request_state_outputs,
            ),
        ),
        linear_1=_bind_tensor(
            linear_1,
            _declare_tensor(
                checkpoint=None,
                checkpoint_key=None,
                reference_key='linear_1',
                layer=None,
                spec=TensorSpec(dtype='float32', shape=(1, image_seq_len, 128)),
                layout=CONTIGUOUS_LAYOUT,
                role=TensorRole.ACTIVATION,
                memory=MemoryClass.FRAME_WORKSPACE,
                lifetime=TensorLifetime.FRAME,
                request_state='linear_1' in request_state_outputs,
            ),
        ),
    )
    bind_logical_tensor_names(tensors, prefix)
    _bind_alias_source(tensors.getitem, tensors.unsqueeze)
    _bind_alias_source(tensors.getitem_1, tensors.unsqueeze_1)
    return tensors


_F16_TENSOR_NAMES = frozenset(())
_F16_TENSOR_PREFIXES = ('',)
_Q6_TENSOR_NAMES = frozenset(())
_Q6_TENSOR_PREFIXES = ()
_Q8_TENSOR_NAMES = frozenset(())
_Q8_TENSOR_PREFIXES = ()


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
