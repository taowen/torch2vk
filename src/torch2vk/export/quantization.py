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

    q8_tensor_names: frozenset[str] = frozenset()
    q8_tensor_prefixes: tuple[str, ...] = ()

    def declare(
        self,
        *,
        checkpoint_key: str,
        dtype: str,
        shape: tuple[int, ...],
    ) -> QuantizedWeightDeclaration:
        dtype = _checkpoint_float_dtype(dtype)
        if dtype != "float32" or len(shape) != 2:
            return QuantizedWeightDeclaration(
                dtype=dtype,
                shape=shape,
                layout_source="CONTIGUOUS_LAYOUT",
            )

        n, k = shape
        force_q8 = checkpoint_key in self.q8_tensor_names or checkpoint_key.startswith(self.q8_tensor_prefixes)
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
