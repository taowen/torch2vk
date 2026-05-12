"""Export-time tensor declarations for GGUF quantized weights."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuantizedWeightDeclaration:
    dtype: str
    shape: tuple[int, ...]
    layout_source: str


@dataclass(frozen=True, slots=True)
class Q4KMWeightQuantization:
    """Match the GGUF Q4_K_M writer's physical tensor layout."""

    q6_tensor_names: frozenset[str] = frozenset()
    q6_tensor_prefixes: tuple[str, ...] = ()
    q8_tensor_names: frozenset[str] = frozenset()
    q8_tensor_prefixes: tuple[str, ...] = ()

    @property
    def has_q6(self) -> bool:
        return bool(self.q6_tensor_names or self.q6_tensor_prefixes)

    def declare(
        self,
        *,
        checkpoint_key: str,
        dtype: str,
        shape: tuple[int, ...],
    ) -> QuantizedWeightDeclaration:
        dtype = _checkpoint_float_dtype(dtype)
        if dtype != "float32":
            return QuantizedWeightDeclaration(
                dtype=dtype,
                shape=shape,
                layout_source="CONTIGUOUS_LAYOUT",
            )

        force_q6 = checkpoint_key in self.q6_tensor_names or checkpoint_key.startswith(self.q6_tensor_prefixes)
        force_q8 = checkpoint_key in self.q8_tensor_names or checkpoint_key.startswith(self.q8_tensor_prefixes)
        if force_q6 and len(shape) >= 2:
            n, k = _matrix_shape(shape)
            if k % 256 != 0:
                raise ValueError(f"Q6_K tensor {checkpoint_key} requires K to be divisible by 256, got {k}")
            return QuantizedWeightDeclaration(
                dtype="uint16",
                shape=(n, k // 256 * 105),
                layout_source=f"q6_k_halfwords_layout(logical_k={k})",
            )
        if force_q8 and len(shape) >= 2:
            n, k = _matrix_shape(shape)
            padded_k = _round_up(k, 32)
            return QuantizedWeightDeclaration(
                dtype="uint16",
                shape=(n, padded_k // 32 * 17),
                layout_source=f"q8_0_halfwords_layout(logical_k={k})",
            )

        if len(shape) != 2:
            return QuantizedWeightDeclaration(
                dtype=dtype,
                shape=shape,
                layout_source="CONTIGUOUS_LAYOUT",
            )

        n, k = shape
        if force_q8 or k % 256 != 0:
            if k % 32 != 0:
                return QuantizedWeightDeclaration(
                    dtype=dtype,
                    shape=shape,
                    layout_source="CONTIGUOUS_LAYOUT",
                )
            return QuantizedWeightDeclaration(
                dtype="uint16",
                shape=(n, k // 32 * 17),
                layout_source=f"q8_0_halfwords_layout(logical_k={k})",
            )

        return QuantizedWeightDeclaration(
            dtype="uint32",
            shape=(n, k // 256 * 36),
            layout_source=f"q4_k_words_layout(logical_k={k})",
        )


def _checkpoint_float_dtype(dtype: str) -> str:
    if dtype in {"float32", "float16", "bfloat16"}:
        return "float32"
    return dtype


def _matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    rows = shape[0]
    cols = 1
    for dim in shape[1:]:
        cols *= dim
    return rows, cols


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
