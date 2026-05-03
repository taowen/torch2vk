"""PyTorch probe transforms for Qwen3 logical tensors."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def last_token_argmax_i32(artifacts: Mapping[str, torch.Tensor]) -> torch.Tensor:
    logits = artifacts["output.logits"]
    return logits[:, -1].argmax(dim=-1).to(torch.int32).contiguous()


QWEN3_PROBE_TRANSFORMS = {
    "last_token_argmax_i32": last_token_argmax_i32,
}
