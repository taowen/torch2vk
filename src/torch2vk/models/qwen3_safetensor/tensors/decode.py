"""Qwen3 decode LogicalTensor tree."""

from __future__ import annotations

from dataclasses import dataclass

from torch2vk.logical import (
    ComparePolicy,
    LogicalTensor,
    MemoryPolicy,
    PyTorchProbe,
    TensorRole,
    activation_tensor,
    input_tensor,
    output_tensor,
)
from torch2vk.models.qwen3_safetensor.spec import Qwen3Spec
from torch2vk.models.qwen3_safetensor.tensors.prefill import (
    Qwen3LayerTensors,
    qwen3_argmax_chunks,
    qwen3_layer_tensors,
)


@dataclass(frozen=True, slots=True)
class Qwen3DecodeTensors:
    input_ids: LogicalTensor
    position_ids: LogicalTensor
    row_indices: LogicalTensor
    rope_freq_factors_placeholder: LogicalTensor
    attention_mask: LogicalTensor
    hidden: LogicalTensor
    layers: tuple[Qwen3LayerTensors, ...]
    final_norm: LogicalTensor
    logits: LogicalTensor
    argmax_partial_values: LogicalTensor
    argmax_partial_indices: LogicalTensor
    next_token_id: LogicalTensor


def qwen3_decode_tensors(
    *,
    batch: int,
    spec: Qwen3Spec,
    max_seq_len: int,
    step_index: int,
) -> Qwen3DecodeTensors:
    if step_index < 0 or step_index >= max_seq_len:
        raise ValueError(f"decode step_index {step_index} outside max_seq_len {max_seq_len}")
    steps = 1
    hidden = activation_tensor(
        "decode.embedding",
        dtype="float32",
        shape=(batch, steps, spec.hidden_size),
    )
    layers: list[Qwen3LayerTensors] = []
    layer_input = hidden
    for layer_index in range(spec.num_hidden_layers):
        layer = qwen3_layer_tensors(
            layer_index=layer_index,
            batch=batch,
            steps=steps,
            cache_steps=max_seq_len,
            spec=spec,
            layer_input=layer_input,
        )
        layers.append(layer)
        layer_input = layer.output
    return Qwen3DecodeTensors(
        input_ids=input_tensor("input.input_ids", dtype="int32", shape=(batch, steps)),
        position_ids=input_tensor("input.position_ids", dtype="int32", shape=(steps,)),
        row_indices=input_tensor("input.row_indices", dtype="int64", shape=(steps,)),
        rope_freq_factors_placeholder=activation_tensor(
            "input.rope_freq_factors_placeholder",
            dtype="float32",
            shape=(spec.head_dim,),
            role=TensorRole.SCRATCH,
        ),
        attention_mask=activation_tensor(
            "input.attention_mask_f16",
            dtype="float16",
            shape=(batch, 1, steps, max_seq_len),
            role=TensorRole.MASK,
            memory=MemoryPolicy.DEVICE_LOCAL,
        ),
        hidden=hidden,
        layers=tuple(layers),
        final_norm=activation_tensor(
            "decode.final_norm",
            dtype="float32",
            shape=(batch, steps, spec.hidden_size),
        ),
        logits=output_tensor(
            "output.logits",
            dtype="float32",
            shape=(batch, steps, spec.vocab_size),
            pytorch_probe=PyTorchProbe(
                kind="manual",
                source="logits",
                normalize="float32_contiguous",
            ),
            compare=ComparePolicy(kind="tensor", rtol=0.0, atol=0.5),
        ),
        argmax_partial_values=activation_tensor(
            "sample.argmax.partial_values",
            dtype="float32",
            shape=(batch, qwen3_argmax_chunks(spec.vocab_size)),
            role=TensorRole.SCRATCH,
        ),
        argmax_partial_indices=activation_tensor(
            "sample.argmax.partial_indices",
            dtype="int32",
            shape=(batch, qwen3_argmax_chunks(spec.vocab_size)),
            role=TensorRole.SCRATCH,
        ),
        next_token_id=output_tensor(
            "output.next_token_id",
            dtype="int32",
            shape=(batch,),
            pytorch_probe=PyTorchProbe(
                kind="derived",
                inputs=("output.logits",),
                transform="last_token_argmax_i32",
                normalize="int32_contiguous",
            ),
            compare=ComparePolicy(kind="token"),
        ),
    )
