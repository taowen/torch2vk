"""Stable tensor metadata types shared across the codebase."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, TypeAlias


Dim: TypeAlias = int | str
TensorShape: TypeAlias = tuple[Dim, ...]


class Residency(StrEnum):
    DEVICE = "device"
    HOST = "host"


@dataclass(frozen=True, slots=True)
class ContiguousLayout:
    kind: str = "contiguous"


@dataclass(frozen=True, slots=True)
class AnyStridedLayout:
    kind: str = "any_strided"


@dataclass(frozen=True, slots=True)
class StridedLayout:
    strides: tuple[Dim, ...]
    kind: str = "strided"


@dataclass(frozen=True, slots=True)
class HeadPackedQGateRowsLayout:
    head_count: Dim
    head_dim: Dim
    parts: tuple[str, str] = ("query", "gate")


@dataclass(frozen=True, slots=True)
class QkvPackedRowsLayout:
    key_dim: Dim
    value_dim: Dim
    parts: tuple[str, str, str] = ("query", "key", "value")


@dataclass(frozen=True, slots=True)
class QkvPackedConv1dChannelsLayout:
    key_dim: Dim
    value_dim: Dim
    kernel_size: Dim
    parts: tuple[str, str, str] = ("query", "key", "value")


@dataclass(frozen=True, slots=True)
class Q8_1X4Layout:
    logical_k: Dim
    block_size: int = 128
    words_per_block: int = 36


@dataclass(frozen=True, slots=True)
class Q4KWordsLayout:
    logical_k: Dim
    block_size: int = 256
    words_per_block: int = 36


@dataclass(frozen=True, slots=True)
class Q6KHalfwordsLayout:
    logical_k: Dim
    block_size: int = 256
    halfwords_per_block: int = 105


TensorLayout: TypeAlias = (
    ContiguousLayout
    | AnyStridedLayout
    | StridedLayout
    | HeadPackedQGateRowsLayout
    | QkvPackedRowsLayout
    | QkvPackedConv1dChannelsLayout
    | Q8_1X4Layout
    | Q4KWordsLayout
    | Q6KHalfwordsLayout
)

CONTIGUOUS_LAYOUT = ContiguousLayout()
ANY_STRIDED_LAYOUT = AnyStridedLayout()


def strided_layout(*strides: Dim) -> StridedLayout:
    return StridedLayout(strides=tuple(strides))


def head_packed_qgate_rows_layout(*, head_count: Dim, head_dim: Dim) -> HeadPackedQGateRowsLayout:
    return HeadPackedQGateRowsLayout(head_count=head_count, head_dim=head_dim)


def qkv_packed_rows_layout(*, key_dim: Dim, value_dim: Dim) -> QkvPackedRowsLayout:
    return QkvPackedRowsLayout(key_dim=key_dim, value_dim=value_dim)


def qkv_packed_conv1d_channels_layout(*, key_dim: Dim, value_dim: Dim, kernel_size: Dim) -> QkvPackedConv1dChannelsLayout:
    return QkvPackedConv1dChannelsLayout(key_dim=key_dim, value_dim=value_dim, kernel_size=kernel_size)


def q8_1_x4_layout(*, logical_k: Dim, block_size: int = 128, words_per_block: int = 36) -> Q8_1X4Layout:
    return Q8_1X4Layout(
        logical_k=logical_k,
        block_size=block_size,
        words_per_block=words_per_block,
    )


def q4_k_words_layout(*, logical_k: Dim, block_size: int = 256, words_per_block: int = 36) -> Q4KWordsLayout:
    return Q4KWordsLayout(
        logical_k=logical_k,
        block_size=block_size,
        words_per_block=words_per_block,
    )


def q6_k_halfwords_layout(
    *,
    logical_k: Dim,
    block_size: int = 256,
    halfwords_per_block: int = 105,
) -> Q6KHalfwordsLayout:
    return Q6KHalfwordsLayout(
        logical_k=logical_k,
        block_size=block_size,
        halfwords_per_block=halfwords_per_block,
    )


def tensor_layout_symbol_names(layout: TensorLayout) -> tuple[str, ...]:
    if isinstance(layout, ContiguousLayout):
        return ()
    if isinstance(layout, AnyStridedLayout):
        return ()
    symbols: list[str] = []
    if isinstance(layout, StridedLayout):
        for stride in layout.strides:
            _append_dim_symbol(symbols, stride)
    elif isinstance(layout, HeadPackedQGateRowsLayout):
        _append_dim_symbol(symbols, layout.head_count)
        _append_dim_symbol(symbols, layout.head_dim)
    elif isinstance(layout, QkvPackedRowsLayout):
        _append_dim_symbol(symbols, layout.key_dim)
        _append_dim_symbol(symbols, layout.value_dim)
    elif isinstance(layout, QkvPackedConv1dChannelsLayout):
        _append_dim_symbol(symbols, layout.key_dim)
        _append_dim_symbol(symbols, layout.value_dim)
        _append_dim_symbol(symbols, layout.kernel_size)
    elif isinstance(layout, Q8_1X4Layout):
        _append_dim_symbol(symbols, layout.logical_k)
    elif isinstance(layout, Q4KWordsLayout):
        _append_dim_symbol(symbols, layout.logical_k)
    else:
        _append_dim_symbol(symbols, layout.logical_k)
    return tuple(symbols)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    dtype: str
    shape: TensorShape
    residency: Residency = Residency.DEVICE

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def with_shape(self, *shape: Dim) -> "TensorSpec":
        return TensorSpec(
            dtype=self.dtype,
            shape=shape,
            residency=self.residency,
        )

    def with_residency(self, residency: Residency) -> "TensorSpec":
        return TensorSpec(
            dtype=self.dtype,
            shape=self.shape,
            residency=residency,
        )


_DTYPE_NBYTES: dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
}


def dtype_nbytes(dtype: str) -> int:
    try:
        return _DTYPE_NBYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype: {dtype}") from exc


def concrete_numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def concrete_nbytes(*, dtype: str, shape: tuple[int, ...]) -> int:
    return concrete_numel(shape) * dtype_nbytes(dtype)


def concrete_shape(spec: TensorSpec) -> tuple[int, ...]:
    shape: list[int] = []
    for dim in spec.shape:
        if not isinstance(dim, int):
            raise ValueError(f"Expected concrete tensor shape, got {spec.shape}")
        shape.append(dim)
    return tuple(shape)


def tensor_nbytes(spec: TensorSpec) -> int:
    return concrete_nbytes(dtype=spec.dtype, shape=concrete_shape(spec))


def resolve_tensor_layout(
    layout: TensorLayout,
    shape_symbols: Mapping[str, int] | None = None,
) -> TensorLayout:
    if isinstance(layout, ContiguousLayout):
        return CONTIGUOUS_LAYOUT
    if isinstance(layout, AnyStridedLayout):
        return ANY_STRIDED_LAYOUT
    if isinstance(layout, StridedLayout):
        return StridedLayout(
            strides=tuple(_resolve_layout_dim(stride, shape_symbols) for stride in layout.strides),
        )
    if isinstance(layout, HeadPackedQGateRowsLayout):
        return HeadPackedQGateRowsLayout(
            head_count=_resolve_layout_dim(layout.head_count, shape_symbols),
            head_dim=_resolve_layout_dim(layout.head_dim, shape_symbols),
            parts=layout.parts,
        )
    if isinstance(layout, QkvPackedRowsLayout):
        return QkvPackedRowsLayout(
            key_dim=_resolve_layout_dim(layout.key_dim, shape_symbols),
            value_dim=_resolve_layout_dim(layout.value_dim, shape_symbols),
            parts=layout.parts,
        )
    if isinstance(layout, QkvPackedConv1dChannelsLayout):
        return QkvPackedConv1dChannelsLayout(
            key_dim=_resolve_layout_dim(layout.key_dim, shape_symbols),
            value_dim=_resolve_layout_dim(layout.value_dim, shape_symbols),
            kernel_size=_resolve_layout_dim(layout.kernel_size, shape_symbols),
            parts=layout.parts,
        )
    if isinstance(layout, Q4KWordsLayout):
        return Q4KWordsLayout(
            logical_k=_resolve_layout_dim(layout.logical_k, shape_symbols),
            block_size=layout.block_size,
            words_per_block=layout.words_per_block,
        )
    if isinstance(layout, Q6KHalfwordsLayout):
        return Q6KHalfwordsLayout(
            logical_k=_resolve_layout_dim(layout.logical_k, shape_symbols),
            block_size=layout.block_size,
            halfwords_per_block=layout.halfwords_per_block,
        )
    return Q8_1X4Layout(
        logical_k=_resolve_layout_dim(layout.logical_k, shape_symbols),
        block_size=layout.block_size,
        words_per_block=layout.words_per_block,
    )


def tensor_layout_matches(
    expected: TensorLayout,
    actual: TensorLayout,
    shape_symbols: Mapping[str, int] | None = None,
) -> bool:
    if isinstance(expected, AnyStridedLayout):
        return isinstance(actual, ContiguousLayout | StridedLayout)
    if isinstance(expected, StridedLayout):
        if not isinstance(actual, StridedLayout):
            return False
        return resolve_tensor_layout(expected, shape_symbols) == resolve_tensor_layout(actual, shape_symbols)
    return resolve_tensor_layout(expected, shape_symbols) == resolve_tensor_layout(actual, shape_symbols)


def bind_tensor_layout_symbols(
    expected: TensorLayout,
    actual: TensorLayout,
    shape_symbols: dict[str, int],
) -> None:
    if isinstance(expected, ContiguousLayout):
        if isinstance(actual, ContiguousLayout):
            return
        raise ValueError(f"Expected contiguous layout, got {actual}")
    if isinstance(expected, AnyStridedLayout):
        if isinstance(actual, ContiguousLayout | StridedLayout):
            return
        raise ValueError(f"Expected strided-capable layout, got {actual}")
    if isinstance(expected, StridedLayout):
        if not isinstance(actual, StridedLayout):
            raise ValueError(f"Expected explicit strided layout, got {actual}")
        if len(expected.strides) != len(actual.strides):
            raise ValueError(
                f"Expected {len(expected.strides)} stride dims, got {len(actual.strides)}"
            )
        for expected_stride, actual_stride in zip(expected.strides, actual.strides, strict=True):
            _bind_layout_symbol(expected_stride, actual_stride, shape_symbols)
        return
    if isinstance(expected, HeadPackedQGateRowsLayout):
        if not isinstance(actual, HeadPackedQGateRowsLayout):
            raise ValueError(f"Expected head-packed q|gate row layout, got {actual}")
        _bind_layout_symbol(expected.head_count, actual.head_count, shape_symbols)
        _bind_layout_symbol(expected.head_dim, actual.head_dim, shape_symbols)
        if expected.parts != actual.parts:
            raise ValueError(f"Expected packed parts {expected.parts}, got {actual.parts}")
        return
    if isinstance(expected, QkvPackedRowsLayout):
        if not isinstance(actual, QkvPackedRowsLayout):
            raise ValueError(f"Expected q|k|v packed row layout, got {actual}")
        _bind_layout_symbol(expected.key_dim, actual.key_dim, shape_symbols)
        _bind_layout_symbol(expected.value_dim, actual.value_dim, shape_symbols)
        if expected.parts != actual.parts:
            raise ValueError(f"Expected packed parts {expected.parts}, got {actual.parts}")
        return
    if isinstance(expected, QkvPackedConv1dChannelsLayout):
        if not isinstance(actual, QkvPackedConv1dChannelsLayout):
            raise ValueError(f"Expected q|k|v packed conv1d channel layout, got {actual}")
        _bind_layout_symbol(expected.key_dim, actual.key_dim, shape_symbols)
        _bind_layout_symbol(expected.value_dim, actual.value_dim, shape_symbols)
        _bind_layout_symbol(expected.kernel_size, actual.kernel_size, shape_symbols)
        if expected.parts != actual.parts:
            raise ValueError(f"Expected packed parts {expected.parts}, got {actual.parts}")
        return
    if isinstance(expected, Q4KWordsLayout):
        if not isinstance(actual, Q4KWordsLayout):
            raise ValueError(f"Expected q4_k row-word layout, got {actual}")
        _bind_layout_symbol(expected.logical_k, actual.logical_k, shape_symbols)
        if expected.block_size != actual.block_size:
            raise ValueError(f"Expected q4_k block size {expected.block_size}, got {actual.block_size}")
        if expected.words_per_block != actual.words_per_block:
            raise ValueError(
                f"Expected q4_k words_per_block {expected.words_per_block}, got {actual.words_per_block}"
            )
        return
    if isinstance(expected, Q6KHalfwordsLayout):
        if not isinstance(actual, Q6KHalfwordsLayout):
            raise ValueError(f"Expected q6_k row-halfword layout, got {actual}")
        _bind_layout_symbol(expected.logical_k, actual.logical_k, shape_symbols)
        if expected.block_size != actual.block_size:
            raise ValueError(f"Expected q6_k block size {expected.block_size}, got {actual.block_size}")
        if expected.halfwords_per_block != actual.halfwords_per_block:
            raise ValueError(
                "Expected q6_k halfwords_per_block "
                f"{expected.halfwords_per_block}, got {actual.halfwords_per_block}"
            )
        return
    if not isinstance(actual, Q8_1X4Layout):
        raise ValueError(f"Expected q8_1_x4 packed layout, got {actual}")
    _bind_layout_symbol(expected.logical_k, actual.logical_k, shape_symbols)
    if expected.block_size != actual.block_size:
        raise ValueError(f"Expected q8_1_x4 block size {expected.block_size}, got {actual.block_size}")
    if expected.words_per_block != actual.words_per_block:
        raise ValueError(
            f"Expected q8_1_x4 words_per_block {expected.words_per_block}, got {actual.words_per_block}"
        )


def validate_tensor_layout(layout: TensorLayout, shape: TensorShape) -> None:
    if isinstance(layout, ContiguousLayout):
        return
    if isinstance(layout, AnyStridedLayout):
        return
    if isinstance(layout, StridedLayout):
        if len(layout.strides) != len(shape):
            raise ValueError(
                f"Strided layout expects {len(shape)} stride dims for shape {shape}, got {len(layout.strides)}"
            )
        for stride in layout.strides:
            concrete_stride = _concrete_dim(stride)
            if concrete_stride is not None and concrete_stride <= 0:
                raise ValueError(f"Strided layout expects positive strides, got {concrete_stride}")
        return
    if isinstance(layout, HeadPackedQGateRowsLayout):
        if len(shape) != 2:
            raise ValueError(f"Head-packed q|gate row layout requires rank-2 tensor shape, got {shape}")

        row_count = _concrete_dim(shape[0])
        head_count = _concrete_dim(layout.head_count)
        head_dim = _concrete_dim(layout.head_dim)
        if row_count is None or head_count is None or head_dim is None:
            return
        expected_row_count = head_count * head_dim * 2
        if row_count != expected_row_count:
            raise ValueError(
                f"Head-packed q|gate row layout expects first dimension {expected_row_count}, got {row_count}"
            )
        return
    if isinstance(layout, QkvPackedRowsLayout):
        if len(shape) != 2:
            raise ValueError(f"Q|K|V packed row layout requires rank-2 tensor shape, got {shape}")
        row_count = _concrete_dim(shape[0])
        key_dim = _concrete_dim(layout.key_dim)
        value_dim = _concrete_dim(layout.value_dim)
        if row_count is None or key_dim is None or value_dim is None:
            return
        expected_row_count = (key_dim * 2) + value_dim
        if row_count != expected_row_count:
            raise ValueError(
                f"Q|K|V packed row layout expects first dimension {expected_row_count}, got {row_count}"
            )
        return

    if isinstance(layout, QkvPackedConv1dChannelsLayout):
        if len(shape) != 3:
            raise ValueError(f"Q|K|V packed conv1d layout requires rank-3 tensor shape, got {shape}")
        channel_count = _concrete_dim(shape[0])
        singleton_dim = _concrete_dim(shape[1])
        kernel_size = _concrete_dim(shape[2])
        key_dim = _concrete_dim(layout.key_dim)
        value_dim = _concrete_dim(layout.value_dim)
        expected_kernel_size = _concrete_dim(layout.kernel_size)
        if singleton_dim is not None and singleton_dim != 1:
            raise ValueError(f"Q|K|V packed conv1d layout expects middle dimension 1, got {singleton_dim}")
        if (
            channel_count is None
            or kernel_size is None
            or key_dim is None
            or value_dim is None
            or expected_kernel_size is None
        ):
            return
        expected_channel_count = (key_dim * 2) + value_dim
        if channel_count != expected_channel_count:
            raise ValueError(
                f"Q|K|V packed conv1d layout expects first dimension {expected_channel_count}, got {channel_count}"
            )
        if kernel_size != expected_kernel_size:
            raise ValueError(
                f"Q|K|V packed conv1d layout expects kernel size {expected_kernel_size}, got {kernel_size}"
            )
        return
    if isinstance(layout, Q4KWordsLayout):
        if len(shape) != 2:
            raise ValueError(f"Q4_K row-word layout requires rank-2 tensor shape, got {shape}")
        word_count = _concrete_dim(shape[1])
        logical_k = _concrete_dim(layout.logical_k)
        if word_count is None or logical_k is None:
            return
        expected_word_count = ((logical_k + layout.block_size - 1) // layout.block_size) * layout.words_per_block
        if word_count != expected_word_count:
            raise ValueError(
                f"Q4_K row-word layout expects second dimension {expected_word_count}, got {word_count}"
            )
        return
    if isinstance(layout, Q6KHalfwordsLayout):
        if len(shape) != 2:
            raise ValueError(f"Q6_K row-halfword layout requires rank-2 tensor shape, got {shape}")
        halfword_count = _concrete_dim(shape[1])
        logical_k = _concrete_dim(layout.logical_k)
        if halfword_count is None or logical_k is None:
            return
        expected_halfword_count = ((logical_k + layout.block_size - 1) // layout.block_size) * layout.halfwords_per_block
        if halfword_count != expected_halfword_count:
            raise ValueError(
                "Q6_K row-halfword layout expects second dimension "
                f"{expected_halfword_count}, got {halfword_count}"
            )
        return

    if len(shape) != 4:
        raise ValueError(f"Q8_1_x4 packed layout requires rank-4 tensor shape, got {shape}")
    block_count = _concrete_dim(shape[2])
    words_per_block = _concrete_dim(shape[3])
    logical_k = _concrete_dim(layout.logical_k)
    if words_per_block is not None and words_per_block != layout.words_per_block:
        raise ValueError(
            f"Q8_1_x4 packed layout expects trailing dimension {layout.words_per_block}, got {words_per_block}"
        )
    if block_count is None or logical_k is None:
        return
    expected_block_count = (logical_k + layout.block_size - 1) // layout.block_size
    if block_count != expected_block_count:
        raise ValueError(
            f"Q8_1_x4 packed layout expects block count {expected_block_count}, got {block_count}"
        )


def _resolve_layout_dim(dim: Dim, shape_symbols: Mapping[str, int] | None) -> int:
    if isinstance(dim, int):
        return dim
    if shape_symbols is None:
        raise KeyError(f"Cannot resolve symbolic layout dim {dim!r} without shape symbols")
    try:
        return int(shape_symbols[dim])
    except KeyError as exc:
        raise KeyError(f"Unresolved symbolic layout dim {dim!r}") from exc


def _append_dim_symbol(symbols: list[str], dim: Dim) -> None:
    if isinstance(dim, str):
        symbols.append(dim)


def _concrete_dim(dim: Dim) -> int | None:
    if isinstance(dim, int):
        return dim
    return None


def _bind_layout_symbol(expected: Dim, actual: Dim, shape_symbols: dict[str, int]) -> None:
    if isinstance(actual, str):
        raise ValueError(f"Actual tensor layout must be concrete, got symbolic dim {actual!r}")
    if isinstance(expected, int):
        if expected != actual:
            raise ValueError(f"Expected concrete layout dim {expected}, got {actual}")
        return
    previous = shape_symbols.get(expected)
    if previous is None:
        shape_symbols[expected] = actual
    elif previous != actual:
        raise ValueError(f"Layout symbol {expected!r} resolved to both {previous} and {actual}")


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    strides = [1] * len(shape)
    running = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = running
        running *= int(shape[axis])
    return tuple(strides)


def tensor_physical_strides(
    layout: TensorLayout,
    shape: tuple[int, ...],
    shape_symbols: Mapping[str, int] | None = None,
) -> tuple[int, ...]:
    if isinstance(layout, ContiguousLayout):
        return contiguous_strides(shape)
    if isinstance(layout, StridedLayout):
        if len(layout.strides) != len(shape):
            raise ValueError(
                f"Strided layout expects {len(shape)} stride dims for shape {shape}, got {len(layout.strides)}"
            )
        return tuple(_resolve_layout_dim(stride, shape_symbols) for stride in layout.strides)
    if isinstance(layout, AnyStridedLayout):
        raise ValueError("Wildcard strided layout does not define concrete physical strides")
    raise ValueError(f"Packed layout {layout} does not expose generic physical strides")
