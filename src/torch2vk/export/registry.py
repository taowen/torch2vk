"""Maps aten op targets to shader variant factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch.fx import Node

from torch2vk.export.shaders import (
    make_add_variant,
    make_argmax_variant,
    make_cat_variant,
    make_conv2d_variant,
    make_embedding_variant,
    make_gelu_variant,
    make_layer_norm_variant,
    make_linear_bias_variant,
    make_linear_nobias_variant,
    make_max_variant,
    make_mean_dim_variant,
    make_mul_variant,
    make_neg_variant,
    make_pow_scalar_variant,
    make_rsqrt_variant,
    make_sdpa_variant,
    make_silu_variant,
    make_slice_variant,
    make_sub_variant,
    make_transpose_variant,
)
from torch2vk.runtime.shader import ShaderVariant


@dataclass(frozen=True, slots=True)
class ShaderBinding:
    target: str
    factory: Callable[[Node], ShaderVariant | None]


class ShaderRegistry:
    def __init__(self, bindings: list[ShaderBinding]) -> None:
        self._bindings = bindings

    def resolve(self, node: Node) -> ShaderVariant | None:
        target = str(node.target)
        for binding in self._bindings:
            if binding.target != target:
                continue
            return binding.factory(node)
        return None


def _make_linear_variant(node: Node) -> ShaderVariant | None:
    has_bias = len(node.args) >= 3 and isinstance(node.args[2], Node)
    if has_bias:
        return make_linear_bias_variant(node)
    return make_linear_nobias_variant(node)


DEFAULT_REGISTRY = ShaderRegistry([
    ShaderBinding("aten.linear.default", _make_linear_variant),
    ShaderBinding("aten.mul.Tensor", make_mul_variant),
    ShaderBinding("aten.add.Tensor", make_add_variant),
    ShaderBinding("aten.sub.Tensor", make_sub_variant),
    ShaderBinding("aten.pow.Tensor_Scalar", make_pow_scalar_variant),
    ShaderBinding("aten.mean.dim", make_mean_dim_variant),
    ShaderBinding("aten.rsqrt.default", make_rsqrt_variant),
    ShaderBinding("aten.silu.default", make_silu_variant),
    ShaderBinding("aten.gelu.default", make_gelu_variant),
    ShaderBinding("aten.neg.default", make_neg_variant),
    ShaderBinding("aten.cat.default", make_cat_variant),
    ShaderBinding("aten.slice.Tensor", make_slice_variant),
    ShaderBinding("aten.scaled_dot_product_attention.default", make_sdpa_variant),
    ShaderBinding("aten.transpose.int", make_transpose_variant),
    ShaderBinding("aten.layer_norm.default", make_layer_norm_variant),
    ShaderBinding("aten.max.default", make_max_variant),
    ShaderBinding("aten.embedding.default", make_embedding_variant),
    ShaderBinding("aten.conv2d.default", make_conv2d_variant),
    ShaderBinding("aten.argmax.default", make_argmax_variant),
])
