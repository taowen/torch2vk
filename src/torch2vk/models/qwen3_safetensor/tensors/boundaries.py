"""Qwen3 end-to-end debug boundary order."""

from __future__ import annotations

from torch2vk.e2e_debug import DebugBoundary
from torch2vk.logical import ComparePolicy


def qwen3_debug_boundaries(*, layers: int) -> tuple[DebugBoundary, ...]:
    return (
        DebugBoundary("qwen3.embedding", 100, ComparePolicy(kind="tensor")),
        *tuple(
            DebugBoundary(
                f"qwen3.layer.{layer:02d}.output",
                200 + layer,
                ComparePolicy(kind="tensor"),
            )
            for layer in range(layers)
        ),
        DebugBoundary("qwen3.output.logits", 400, ComparePolicy(kind="tensor")),
        DebugBoundary("qwen3.output.next_token_id", 500, ComparePolicy(kind="token")),
    )
