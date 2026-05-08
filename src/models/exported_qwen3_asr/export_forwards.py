"""Forward functions shared by exported_qwen3_asr export and debug runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import MethodType
from typing import cast

import torch


@contextmanager
def patched_forward(
    module: torch.nn.Module,
    forward: Callable[..., object],
) -> Iterator[None]:
    original_forward = module.forward
    setattr(module, "forward", MethodType(forward, module))
    try:
        yield
    finally:
        setattr(module, "forward", original_forward)


def export_audio_tower_forward(
    self: torch.nn.Module,
    x: torch.Tensor,
    position_embedding: torch.Tensor,
    compact_index: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if attention_mask is None:
        raise ValueError("attention_mask is required")
    x = torch.nn.functional.gelu(self.get_submodule("conv2d1")(x))
    x = torch.nn.functional.gelu(self.get_submodule("conv2d2")(x))
    x = torch.nn.functional.gelu(self.get_submodule("conv2d3")(x))
    batch, channels, freq, time = x.shape
    hidden_states = self.get_submodule("conv_out")(
        x.reshape(batch, channels * freq, time).transpose(1, 2)
    )
    hidden_states = hidden_states + position_embedding
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    hidden_states = torch.index_select(hidden_states, 0, compact_index)

    for layer in cast(torch.nn.ModuleList, self.get_submodule("layers")):
        if cu_seqlens is None:
            hidden_states = _run_audio_layer(layer, hidden_states, attention_mask)
        else:
            hidden_states = layer(
                hidden_states,
                cu_seqlens,
                attention_mask=attention_mask,
            )[0]

    hidden_states = self.get_submodule("ln_post")(hidden_states)
    hidden_states = self.get_submodule("proj1")(hidden_states)
    hidden_states = cast(torch.nn.Module, getattr(self, "act"))(hidden_states)
    last_hidden_state = self.get_submodule("proj2")(hidden_states)
    return {"last_hidden_state": last_hidden_state}


def export_audio_inject_forward(
    self: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    audio_positions: torch.Tensor,
    audio_features: torch.Tensor,
) -> torch.Tensor:
    return torch.index_copy(
        inputs_embeds,
        1,
        audio_positions,
        audio_features.unsqueeze(0),
    )


def _run_audio_layer(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = layer.get_submodule("self_attn_layer_norm")(hidden_states)
    hidden_states = _run_audio_attention(
        layer.get_submodule("self_attn"),
        hidden_states,
        attention_mask,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = layer.get_submodule("final_layer_norm")(hidden_states)
    hidden_states = layer.get_submodule("fc1")(hidden_states)
    hidden_states = cast(torch.nn.Module, getattr(layer, "activation_fn"))(hidden_states)
    hidden_states = layer.get_submodule("fc2")(hidden_states)
    return residual + hidden_states


def _run_audio_attention(
    attention: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    seq_length, _ = hidden_states.size()
    num_heads = int(getattr(attention, "num_heads"))
    query_states = attention.get_submodule("q_proj")(hidden_states).reshape(
        seq_length, num_heads, -1
    )
    key_states = attention.get_submodule("k_proj")(hidden_states).reshape(
        seq_length, num_heads, -1
    )
    value_states = attention.get_submodule("v_proj")(hidden_states).reshape(
        seq_length, num_heads, -1
    )

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    attention_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(getattr(attention, "scaling")),
    )
    attention_output = attention_output.transpose(1, 2).reshape(seq_length, -1)
    return attention.get_submodule("out_proj")(attention_output)
