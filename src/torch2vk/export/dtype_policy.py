"""Dtype policy for exported logical tensors and shader activation storage.

Checkpoint parameters keep their original dtype. Exported floating-point
activations use f16 storage by default, while numerically sensitive scalar
normalization/reduction intermediates stay f32.
"""

from __future__ import annotations

from torch.fx import Node

INTEGER_DTYPES = frozenset(("int64", "int32", "uint32"))
PARAMETER_DTYPES = frozenset(("bfloat16", "float32", "float16", "int64", "int32", "uint32"))
FLOAT32_INTERMEDIATE_TARGETS = frozenset((
    "aten.pow.Tensor_Scalar",
    "aten.mean.dim",
    "aten.rsqrt.default",
))


def logical_tensor_dtype(
    *,
    is_parameter: bool,
    dtype: str,
    force_float32: bool = False,
) -> str:
    if is_parameter:
        if dtype in PARAMETER_DTYPES:
            return dtype
        raise ValueError(f"Unsupported checkpoint parameter dtype: {dtype}")
    if force_float32:
        return "float32"
    return dtype if dtype in INTEGER_DTYPES else "float16"


def requires_float32_intermediate(node: Node) -> bool:
    target = str(node.target)
    if target in FLOAT32_INTERMEDIATE_TARGETS:
        return True
    return target == "aten.add.Tensor" and len(node.args) >= 2 and not isinstance(node.args[1], Node)
