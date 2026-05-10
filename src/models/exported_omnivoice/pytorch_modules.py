"""Tensor-only PyTorch modules used by exported OmniVoice.

These modules are the shared source for torch.export references and Vulkan
debug compare. The outer generation loop stays in Python; each step stage here
is a pure tensor transform.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn

from omnivoice.models.omnivoice import OmniVoice


def torch_tensor(inputs: dict[str, np.ndarray], name: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(inputs[name])).cuda()


class InputEmbedModule(nn.Module):
    def __init__(self, model: OmniVoice) -> None:
        super().__init__()
        self.text_embedding = cast(nn.Module, model.get_input_embeddings())
        self.audio_embedding = cast(nn.Module, model.audio_embeddings)
        self._codebook_layer_offsets = cast(
            torch.Tensor,
            model.codebook_layer_offsets,
        ).detach().clone()

    def forward(self, input_ids: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        audio_mask_bool = audio_mask.to(torch.bool)
        text_embeds = self.text_embedding(input_ids[:, 0, :])
        shifted_ids = (
            input_ids * audio_mask_bool.unsqueeze(1)
        ) + self._codebook_layer_offsets.reshape(1, -1, 1)
        audio_embeds = self.audio_embedding(shifted_ids).sum(dim=1)
        return torch.where(audio_mask_bool.unsqueeze(-1), audio_embeds, text_embeds)


class LlmForwardModule(nn.Module):
    def __init__(self, model: OmniVoice) -> None:
        super().__init__()
        self.layers = cast(nn.ModuleList, model.llm.layers)
        self.norm = cast(nn.Module, model.llm.norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_embeddings = (cos, sin)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
        return self.norm(hidden_states)


class AudioHeadModule(nn.Module):
    def __init__(self, model: OmniVoice) -> None:
        super().__init__()
        self.audio_heads = model.audio_heads

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.audio_heads(hidden_states)


class TokenScoreModule(nn.Module):
    def __init__(
        self,
        *,
        num_audio_codebook: int,
        audio_vocab_size: int,
        guidance_scale: float = 2.0,
        layer_penalty: float = 5.0,
        position_temperature: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_audio_codebook = num_audio_codebook
        self.audio_vocab_size = audio_vocab_size
        self.guidance_scale = guidance_scale
        self.layer_penalty = layer_penalty
        self.position_temperature = position_temperature

    def forward(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        audio_mask_id: torch.Tensor,
        rng_seed: torch.Tensor,
        step_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_len = tokens.shape[-1]
        seq_len = logits.shape[1]
        logits = logits.view(2, seq_len, self.num_audio_codebook, self.audio_vocab_size)
        cond_logits = logits[0:1, seq_len - target_len : seq_len].permute(0, 2, 1, 3)
        uncond_logits = logits[1:2, :target_len].permute(0, 2, 1, 3)

        cond_log_probs = torch.log_softmax(cond_logits, dim=-1)
        uncond_log_probs = torch.log_softmax(uncond_logits, dim=-1)
        guided_logits = cond_log_probs + self.guidance_scale * (
            cond_log_probs - uncond_log_probs
        )
        log_probs = torch.log_softmax(guided_logits, dim=-1)

        vocab = torch.arange(self.audio_vocab_size, device=logits.device).view(1, 1, 1, -1)
        mask_token = audio_mask_id.reshape(1, 1, 1, 1)
        log_probs = torch.where(
            vocab == mask_token,
            torch.full_like(log_probs, -float("inf")),
            log_probs,
        )
        candidate_tokens = torch.argmax(log_probs, dim=-1)[0].to(torch.int64)
        candidate_scores = torch.max(log_probs, dim=-1)[0][0]

        layer_ids = torch.arange(
            self.num_audio_codebook,
            device=logits.device,
            dtype=torch.float32,
        ).view(self.num_audio_codebook, 1)
        candidate_scores = candidate_scores - layer_ids * self.layer_penalty

        if self.position_temperature > 0.0:
            flat_pos = torch.arange(
                self.num_audio_codebook * target_len,
                device=logits.device,
                dtype=torch.int64,
            ).view(self.num_audio_codebook, target_len)
            candidate_scores = candidate_scores + self._gumbel_noise(
                flat_pos,
                rng_seed,
                step_index,
            ) * self.position_temperature

        candidate_scores = torch.where(
            tokens[0] != audio_mask_id.reshape(()),
            torch.full_like(candidate_scores, -torch.finfo(torch.float32).max),
            candidate_scores,
        )
        return candidate_tokens, candidate_scores.to(torch.float32)

    def _gumbel_noise(
        self,
        flat_pos: torch.Tensor,
        rng_seed: torch.Tensor,
        step_index: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.bitwise_xor(
            self._u32(rng_seed.to(torch.int64).reshape(())),
            self._u32(step_index.to(torch.int64).reshape(()) * 0x9E3779B9),
        )
        x = torch.bitwise_xor(x, self._u32(flat_pos.to(torch.int64) * 0x85EBCA6B))
        hashed = self._hash_u32(x)
        u = (torch.bitwise_and(hashed, 0x00FFFFFF).to(torch.float32) + 0.5) * (
            1.0 / 16777216.0
        )
        u = torch.clamp(u, 1.0e-7, 1.0 - 1.0e-7)
        return -torch.log(-torch.log(u))

    def _hash_u32(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
        x = self._u32(x * 0x7FEB352D)
        x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 15))
        x = self._u32(x * 0x846CA68B)
        x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
        return self._u32(x)

    def _u32(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_and(x, 0xFFFFFFFF)


class TokenUpdateModule(nn.Module):
    def forward(
        self,
        tokens: torch.Tensor,
        batch_input_ids: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_scores: torch.Tensor,
        unmask_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_scores = candidate_scores.reshape(-1)
        flat_candidate_tokens = candidate_tokens.reshape(-1)
        flat_tokens = tokens.reshape(-1)
        total = flat_scores.shape[0]
        indices = torch.arange(total, device=flat_scores.device)

        better = (flat_scores[None, :] > flat_scores[:, None]) | (
            (flat_scores[None, :] == flat_scores[:, None])
            & (indices[None, :] < indices[:, None])
        )
        rank = better.to(torch.int64).sum(dim=1)
        selected = rank < unmask_count.to(torch.int64).reshape(())
        updated_flat = torch.where(selected, flat_candidate_tokens, flat_tokens)
        updated_tokens = updated_flat.reshape_as(tokens).to(torch.int64)

        updated_batch_input_ids = batch_input_ids.clone()
        target_len = tokens.shape[-1]
        seq_len = batch_input_ids.shape[-1]
        updated_batch_input_ids[0:1, :, seq_len - target_len : seq_len] = updated_tokens
        updated_batch_input_ids[1:2, :, :target_len] = updated_tokens
        return updated_tokens, updated_batch_input_ids.to(torch.int64)


class InputEmbedReference:
    def __init__(self, model: OmniVoice) -> None:
        self.module = InputEmbedModule(model).float().cuda().eval()

    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        with torch.no_grad():
            hidden_states = self.module(
                torch_tensor(inputs, "input_ids").long(),
                torch_tensor(inputs, "audio_mask").to(torch.bool),
            ).float()
        return {"hidden_states": hidden_states}


class LlmForwardReference:
    def __init__(self, model: OmniVoice) -> None:
        self.module = LlmForwardModule(model).float().cuda().eval()

    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        with torch.no_grad():
            output = self.module(
                torch_tensor(inputs, "hidden_states").float(),
                torch_tensor(inputs, "cos").float(),
                torch_tensor(inputs, "sin").float(),
                torch_tensor(inputs, "attention_mask").float(),
            ).float()
        return {"mul_365": output}


class AudioHeadReference:
    def __init__(self, model: OmniVoice) -> None:
        self.module = AudioHeadModule(model).float().cuda().eval()

    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        with torch.no_grad():
            output = self.module(torch_tensor(inputs, "input").float()).float()
        return {"linear": output}


class TokenScoreReference:
    def __init__(self, model: OmniVoice) -> None:
        self.module = TokenScoreModule(
            num_audio_codebook=model.config.num_audio_codebook,
            audio_vocab_size=model.config.audio_vocab_size,
        ).cuda().eval()

    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        with torch.no_grad():
            candidate_tokens, candidate_scores = self.module(
                torch_tensor(inputs, "logits").float(),
                torch_tensor(inputs, "tokens").long(),
                torch_tensor(inputs, "audio_mask_id").long(),
                torch_tensor(inputs, "rng_seed").long(),
                torch_tensor(inputs, "step_index").long(),
            )
        return {
            "candidate_tokens": candidate_tokens,
            "candidate_scores": candidate_scores,
        }


class TokenUpdateReference:
    def __init__(self) -> None:
        self.module = TokenUpdateModule().cuda().eval()

    def execute(self, inputs: dict[str, np.ndarray]) -> dict[str, object]:
        with torch.no_grad():
            tokens, batch_input_ids = self.module(
                torch_tensor(inputs, "tokens").long(),
                torch_tensor(inputs, "batch_input_ids").long(),
                torch_tensor(inputs, "candidate_tokens").long(),
                torch_tensor(inputs, "candidate_scores").float(),
                torch_tensor(inputs, "unmask_count"),
            )
        return {
            "tokens": tokens,
            "batch_input_ids": batch_input_ids,
        }
