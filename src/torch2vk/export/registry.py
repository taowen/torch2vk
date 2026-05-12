"""Maps aten op targets to shader variant factories.

Historical shader names often keep an ``_f32`` suffix because the shader family
uses f32 arithmetic or accumulation. Activation storage dtype is defined by the
ShaderVariant contracts produced by each factory, not by the name suffix.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch.fx import Node

from torch2vk.export.shaders import (
    make_add_variant,
    make_argmax_variant,
    make_cat_variant,
    make_conv1d_q8_0_variant,
    make_conv1d_variant,
    make_conv2d_q8_0_variant,
    make_conv2d_variant,
    make_conv_transpose1d_q8_0_variant,
    make_conv_transpose1d_variant,
    make_embedding_q4_k_m_variant,
    make_embedding_q8_0_variant,
    make_embedding_variant,
    make_gelu_variant,
    make_index_copy_variant,
    make_index_select_variant,
    make_layer_norm_variant,
    make_linear_bias_q8_0_variant,
    make_linear_bias_variant,
    make_linear_nobias_variant,
    make_linear_nobias_q4_k_m_variant,
    make_linear_nobias_q8_0_variant,
    make_max_variant,
    make_mean_dim_variant,
    make_mul_variant,
    make_neg_variant,
    make_pow_scalar_variant,
    make_permute_variant,
    make_reciprocal_variant,
    make_rsqrt_variant,
    make_sdpa_variant,
    make_select_variant,
    make_silu_variant,
    make_sin_variant,
    make_slice_variant,
    make_sub_variant,
    make_transpose_variant,
)
from torch2vk.runtime.shader import ShaderVariant


@dataclass(frozen=True, slots=True)
class ShaderBinding:
    target: str
    factory: Callable[[Node, str], ShaderVariant | None]


class ShaderRegistry:
    def __init__(self, bindings: list[ShaderBinding], *, activation_dtype: str = "float16") -> None:
        self._bindings = bindings
        self._activation_dtype = activation_dtype

    @property
    def activation_dtype(self) -> str:
        return self._activation_dtype

    def with_activation_dtype(self, activation_dtype: str) -> ShaderRegistry:
        return ShaderRegistry(self._bindings, activation_dtype=activation_dtype)

    def resolve(self, node: Node) -> ShaderVariant | None:
        target = str(node.target)
        for binding in self._bindings:
            if binding.target != target:
                continue
            variant = binding.factory(node, self._activation_dtype)
            if variant is None:
                return None
            return variant
        return None


def _make_linear_variant(node: Node, activation_dtype: str) -> ShaderVariant | None:
    has_bias = len(node.args) >= 3 and isinstance(node.args[2], Node)
    if has_bias:
        return make_linear_bias_variant(node, activation_dtype)
    return make_linear_nobias_variant(node, activation_dtype)


def _make_q4_k_m_linear_variant(node: Node, activation_dtype: str) -> ShaderVariant | None:
    has_bias = len(node.args) >= 3 and isinstance(node.args[2], Node)
    if has_bias:
        return make_linear_bias_variant(node, activation_dtype)
    return make_linear_nobias_q4_k_m_variant(node, activation_dtype)


def _make_q8_0_linear_variant(node: Node, activation_dtype: str) -> ShaderVariant | None:
    has_bias = len(node.args) >= 3 and isinstance(node.args[2], Node)
    if has_bias:
        return make_linear_bias_q8_0_variant(node, activation_dtype)
    return make_linear_nobias_q8_0_variant(node, activation_dtype)


DEFAULT_REGISTRY = ShaderRegistry(
    [
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
        ShaderBinding("aten.permute.default", make_permute_variant),
        ShaderBinding("aten.layer_norm.default", make_layer_norm_variant),
        ShaderBinding("aten.max.default", make_max_variant),
        ShaderBinding("aten.embedding.default", make_embedding_variant),
        ShaderBinding("aten.conv1d.default", make_conv1d_variant),
        ShaderBinding("aten.conv_transpose1d.default", make_conv_transpose1d_variant),
        ShaderBinding("aten.conv2d.default", make_conv2d_variant),
        ShaderBinding("aten.argmax.default", make_argmax_variant),
        ShaderBinding("aten.index_copy.default", make_index_copy_variant),
        ShaderBinding("aten.index_select.default", make_index_select_variant),
        ShaderBinding("aten.select.int", make_select_variant),
        ShaderBinding("aten.sin.default", make_sin_variant),
        ShaderBinding("aten.reciprocal.default", make_reciprocal_variant),
    ]
)

Q4_K_M_REGISTRY = ShaderRegistry(
    [
        ShaderBinding("aten.linear.default", _make_q4_k_m_linear_variant),
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
        ShaderBinding("aten.permute.default", make_permute_variant),
        ShaderBinding("aten.layer_norm.default", make_layer_norm_variant),
        ShaderBinding("aten.max.default", make_max_variant),
        ShaderBinding("aten.embedding.default", make_embedding_q4_k_m_variant),
        ShaderBinding("aten.conv1d.default", make_conv1d_variant),
        ShaderBinding("aten.conv_transpose1d.default", make_conv_transpose1d_variant),
        ShaderBinding("aten.conv2d.default", make_conv2d_variant),
        ShaderBinding("aten.argmax.default", make_argmax_variant),
        ShaderBinding("aten.index_copy.default", make_index_copy_variant),
        ShaderBinding("aten.index_select.default", make_index_select_variant),
        ShaderBinding("aten.select.int", make_select_variant),
        ShaderBinding("aten.sin.default", make_sin_variant),
        ShaderBinding("aten.reciprocal.default", make_reciprocal_variant),
    ]
)

Q8_0_REGISTRY = ShaderRegistry(
    [
        ShaderBinding("aten.linear.default", _make_q8_0_linear_variant),
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
        ShaderBinding("aten.permute.default", make_permute_variant),
        ShaderBinding("aten.layer_norm.default", make_layer_norm_variant),
        ShaderBinding("aten.max.default", make_max_variant),
        ShaderBinding("aten.embedding.default", make_embedding_q8_0_variant),
        ShaderBinding("aten.conv1d.default", make_conv1d_q8_0_variant),
        ShaderBinding("aten.conv_transpose1d.default", make_conv_transpose1d_q8_0_variant),
        ShaderBinding("aten.conv2d.default", make_conv2d_q8_0_variant),
        ShaderBinding("aten.argmax.default", make_argmax_variant),
        ShaderBinding("aten.index_copy.default", make_index_copy_variant),
        ShaderBinding("aten.index_select.default", make_index_select_variant),
        ShaderBinding("aten.select.int", make_select_variant),
        ShaderBinding("aten.sin.default", make_sin_variant),
        ShaderBinding("aten.reciprocal.default", make_reciprocal_variant),
    ]
)
