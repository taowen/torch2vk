"""Qwen3 safetensor debug boundary declarations."""

from __future__ import annotations

from torch2vk.logical import ComparePolicy, LogicalTensor
from torch2vk.schema import BoundaryRule

from .execution import Qwen3ExecutionTensors
from .schema import qwen3_weight_tensors
from .spec import Qwen3Spec


def qwen3_prefill_initial_tensors(
    *,
    spec: Qwen3Spec,
    tensors: Qwen3ExecutionTensors,
) -> tuple[LogicalTensor, ...]:
    return (
        tensors.input_ids,
        tensors.position_ids,
        tensors.row_indices,
        tensors.rope_freq_factors_placeholder,
        tensors.attention_mask,
        *(layer.attention_sinks_placeholder for layer in tensors.layers),
        *qwen3_weight_tensors(spec),
    )


def qwen3_prefill_debug_boundaries(
    tensors: Qwen3ExecutionTensors,
) -> tuple[BoundaryRule, ...]:
    tensor_compare = ComparePolicy(kind="tensor", rtol=1e-2, atol=1e-2)
    token_compare = ComparePolicy(kind="token")
    boundaries: list[BoundaryRule] = [
        BoundaryRule(
            name="prefill.embedding",
            phase="model",
            order=100,
            tensors=(tensors.hidden,),
            compare=tensor_compare,
            checkpoint=tensors.input_ids,
            readback="writer-io",
        )
    ]
    for layer_index, layer in enumerate(tensors.layers):
        order = 200 + layer_index * 100
        boundaries.extend(
            (
                BoundaryRule(
                    name=f"prefill.layer.{layer_index:02d}.attention",
                    phase="model",
                    order=order,
                    tensors=(
                        layer.q_proj,
                        layer.k_proj,
                        layer.v_proj,
                        layer.q_rope,
                        layer.key_cache,
                        layer.value_cache,
                        layer.attention_context,
                        layer.attention_residual,
                    ),
                    compare=tensor_compare,
                    checkpoint=layer.input,
                    readback="writer-io",
                ),
                BoundaryRule(
                    name=f"prefill.layer.{layer_index:02d}.mlp",
                    phase="model",
                    order=order + 50,
                    tensors=(
                        layer.post_attention_norm,
                        layer.mlp_gate,
                        layer.mlp_up,
                        layer.mlp_gated,
                        layer.output,
                    ),
                    compare=tensor_compare,
                    checkpoint=layer.post_attention_norm,
                    readback="writer-io",
                ),
            )
        )
    boundaries.extend(
        (
            BoundaryRule(
                name="prefill.final_norm",
                phase="model",
                order=900,
                tensors=(tensors.final_norm,),
                compare=tensor_compare,
                checkpoint=tensors.layers[-1].output,
                readback="writer-io",
            ),
            BoundaryRule(
                name="prefill.logits",
                phase="model",
                order=1000,
                tensors=(tensors.logits,),
                compare=tensor_compare,
                checkpoint=tensors.final_norm,
                readback="writer-io",
            ),
            BoundaryRule(
                name="prefill.next_token",
                phase="sample",
                order=1100,
                tensors=(tensors.next_token_id,),
                compare=token_compare,
                checkpoint=tensors.logits,
                readback="writer-io",
            ),
        )
    )
    return tuple(boundaries)
