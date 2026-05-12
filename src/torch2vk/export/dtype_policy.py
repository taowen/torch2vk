"""Dtype policy for exported logical tensors and shader activation storage.

Checkpoint parameters keep their original dtype. Exported floating-point
activations use f16 storage by default, while numerically sensitive scalar
normalization/reduction intermediates stay f32.
"""

from __future__ import annotations

from torch.fx import Node

INTEGER_DTYPES = frozenset(("int64", "int32", "uint32"))
PARAMETER_DTYPES = frozenset(("bfloat16", "float32", "float16", "int64", "int32", "uint32"))
FLOAT32_INTERMEDIATE_TARGETS = frozenset(
    (
        "aten.mean.dim",
        "aten.rsqrt.default",
        "aten.reciprocal.default",
    )
)


def logical_tensor_dtype(
    *,
    is_parameter: bool,
    dtype: str,
    activation_dtype: str = "float16",
    force_float32: bool = False,
) -> str:
    if is_parameter:
        if dtype in PARAMETER_DTYPES:
            return dtype
        raise ValueError(f"Unsupported checkpoint parameter dtype: {dtype}")
    if force_float32:
        return "float32"
    return dtype if dtype in INTEGER_DTYPES else activation_dtype


def requires_float32_intermediate(node: Node) -> bool:
    target = str(node.target)
    if target == "aten.pow.Tensor_Scalar":
        return _pow_requires_float32(node)
    if target in FLOAT32_INTERMEDIATE_TARGETS:
        return True
    return target == "aten.add.Tensor" and _scalar_add_feeds_inverse(node)


def _pow_requires_float32(node: Node) -> bool:
    if _is_sin_square(node):
        return False
    return True


def _is_sin_square(node: Node) -> bool:
    if len(node.args) < 2:
        return False
    x = node.args[0]
    exponent = node.args[1]
    return (
        isinstance(x, Node)
        and str(x.target) == "aten.sin.default"
        and isinstance(exponent, (int, float))
        and float(exponent) == 2.0
    )


def _scalar_add_feeds_inverse(node: Node) -> bool:
    if len(node.args) < 2 or isinstance(node.args[1], Node):
        return False
    return any(
        str(user.target) in {"aten.rsqrt.default", "aten.reciprocal.default"} for user in node.users
    )
