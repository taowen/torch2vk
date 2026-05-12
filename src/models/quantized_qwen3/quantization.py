"""Quantization topology for generated Qwen3."""

from __future__ import annotations

Q8_TENSOR_NAMES: tuple[str, ...] = ()
Q6_LAYER_INDICES = (0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 24, 25, 26, 27)
Q6_TENSOR_NAMES = (
    "lm_head.weight",
    *(f"model.layers.{idx}.self_attn.v_proj.weight" for idx in Q6_LAYER_INDICES),
    *(f"model.layers.{idx}.mlp.down_proj.weight" for idx in Q6_LAYER_INDICES),
)
