"""OmniVoice generation-state helpers shared by eager debug loops."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class OmniVoiceSelectionResult:
    pred_tokens: torch.Tensor
    guided_scores: torch.Tensor
    selection_scores: torch.Tensor
    update_mask: torch.Tensor
    tokens_after: torch.Tensor


def omnivoice_unmask_schedule(
    *,
    total: int,
    num_steps: int,
    t_shift: float,
) -> tuple[int, ...]:
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32)
    timesteps = t_shift * timesteps / (1.0 + (t_shift - 1.0) * timesteps)
    remaining = int(total)
    schedule: list[int] = []
    for step in range(num_steps):
        if step == num_steps - 1:
            count = remaining
        else:
            span = float((timesteps[step + 1] - timesteps[step]).item())
            count = min(remaining, math.ceil(total * span))
        schedule.append(count)
        remaining -= count
    return tuple(schedule)


def select_omnivoice_audio_tokens(
    *,
    tokens_before: torch.Tensor,
    cond_logits: torch.Tensor,
    uncond_logits: torch.Tensor,
    unmask_count: int,
    mask_id: int,
    guidance_scale: float,
    layer_penalty_factor: float,
) -> OmniVoiceSelectionResult:
    if tokens_before.ndim != 2:
        raise ValueError(f"tokens_before must be rank 2, got shape {tuple(tokens_before.shape)}")
    if cond_logits.shape != uncond_logits.shape:
        raise ValueError(
            "cond_logits and uncond_logits must have the same shape, "
            f"got {tuple(cond_logits.shape)} and {tuple(uncond_logits.shape)}"
        )
    if cond_logits.ndim != 3:
        raise ValueError(f"logits must be rank 3, got shape {tuple(cond_logits.shape)}")
    codebooks, steps = tokens_before.shape
    if cond_logits.shape[:2] != tokens_before.shape:
        raise ValueError(
            "logits leading dimensions must match tokens_before, "
            f"got {tuple(cond_logits.shape[:2])} and {tuple(tokens_before.shape)}"
        )
    if unmask_count < 0:
        raise ValueError(f"unmask_count must be non-negative, got {unmask_count}")

    cond_log_probs = functional.log_softmax(cond_logits.float(), dim=-1)
    uncond_log_probs = functional.log_softmax(uncond_logits.float(), dim=-1)
    guided_scores = functional.log_softmax(
        cond_log_probs + guidance_scale * (cond_log_probs - uncond_log_probs),
        dim=-1,
    )
    if 0 <= mask_id < guided_scores.shape[-1]:
        guided_scores[..., mask_id] = -torch.inf
    pred_tokens = torch.argmax(guided_scores, dim=-1).to(torch.int32)

    layer_penalty = (
        torch.arange(codebooks, dtype=torch.float32, device=guided_scores.device).view(codebooks, 1)
        * layer_penalty_factor
    )
    selection_scores = torch.max(guided_scores, dim=-1).values - layer_penalty
    selection_scores = torch.where(
        tokens_before.to(torch.int64) == int(mask_id),
        selection_scores,
        torch.full_like(selection_scores, -torch.inf),
    )
    flat_scores = selection_scores.reshape(-1)
    valid = torch.nonzero(torch.isfinite(flat_scores), as_tuple=False).flatten()
    update_mask = torch.zeros((codebooks, steps), dtype=torch.int32, device=tokens_before.device)
    tokens_after = tokens_before.to(torch.int32).clone()
    if valid.numel() == 0 or unmask_count == 0:
        return OmniVoiceSelectionResult(
            pred_tokens=pred_tokens,
            guided_scores=guided_scores,
            selection_scores=selection_scores,
            update_mask=update_mask,
            tokens_after=tokens_after,
        )

    chosen_count = min(int(unmask_count), int(valid.numel()))
    valid_scores = flat_scores.index_select(0, valid)
    topk = torch.topk(valid_scores, k=chosen_count).indices
    chosen = valid.index_select(0, topk)
    flat_tokens = tokens_after.reshape(-1)
    flat_pred = pred_tokens.reshape(-1)
    flat_mask = update_mask.reshape(-1)
    flat_tokens[chosen] = flat_pred.index_select(0, chosen)
    flat_mask[chosen] = 1
    return OmniVoiceSelectionResult(
        pred_tokens=pred_tokens,
        guided_scores=guided_scores,
        selection_scores=selection_scores,
        update_mask=update_mask,
        tokens_after=tokens_after,
    )
