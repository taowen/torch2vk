from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
)


OUTPUT_LAYERS_QWEN3 = (9, 18, 27)
MAX_LENGTH = 512


class _SafeTensorReader:
    def __init__(self, model_dir: Path) -> None:
        index_path = model_dir / "model.safetensors.index.json"
        data = json.loads(index_path.read_text(encoding="utf-8"))
        self.model_dir = model_dir
        self.weight_map: dict[str, str] = data["weight_map"]

    def tensor(self, key: str) -> torch.Tensor:
        filename = self.weight_map[key]
        with safe_open(self.model_dir / filename, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def layer_state(
        self,
        layer_idx: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        prefix = f"model.layers.{layer_idx}."
        state: dict[str, torch.Tensor] = {}
        for key in sorted(self.weight_map):
            if key.startswith(prefix):
                state[key.removeprefix(prefix)] = self.tensor(key).to(device=device, dtype=dtype)
        return state


class Qwen3TextEncoder(nn.Module):
    def __init__(
        self,
        model_dir: str,
        *,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.config = cast(Qwen3Config, AutoConfig.from_pretrained(self.model_dir))
        self.config._attn_implementation = "eager"
        self.max_length = MAX_LENGTH
        self.reader = _SafeTensorReader(self.model_dir)

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> torch.Tensor:
        input_ids = []
        attention_masks = []
        for prompt in prompts:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            input_ids.append(model_inputs["input_ids"])
            attention_masks.append(model_inputs["attention_mask"])

        ids = torch.cat(input_ids, dim=0).to(self.device)
        attention_mask = torch.cat(attention_masks, dim=0).to(self.device)

        embed_weight = self.reader.tensor("model.embed_tokens.weight").to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        hidden_states = F.embedding(ids, embed_weight)
        del embed_weight
        torch.cuda.empty_cache()

        position_ids = torch.arange(hidden_states.shape[1], device=self.device).unsqueeze(0)
        with torch.device("meta"):
            rotary = Qwen3RotaryEmbedding(config=self.config)
        rotary = rotary.to_empty(device=self.device).to(dtype=self.torch_dtype)
        fresh_rotary = Qwen3RotaryEmbedding(config=self.config, device=self.device)
        rotary.inv_freq = fresh_rotary.inv_freq.to(self.device)
        rotary.original_inv_freq = rotary.inv_freq.clone()
        position_embeddings = rotary(hidden_states, position_ids)
        attention = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        del rotary, fresh_rotary
        torch.cuda.empty_cache()

        captures: list[torch.Tensor] = []
        for layer_idx in range(int(self.config.num_hidden_layers)):
            with torch.device("meta"):
                layer = Qwen3DecoderLayer(self.config, layer_idx)
            layer = layer.to_empty(device=self.device).to(dtype=self.torch_dtype).eval()
            state = self.reader.layer_state(
                layer_idx,
                device=self.device,
                dtype=self.torch_dtype,
            )
            layer.load_state_dict(state, strict=True, assign=True)
            output = layer(
                hidden_states,
                attention_mask=attention,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            hidden_states = output[0] if isinstance(output, tuple) else output
            if layer_idx + 1 in OUTPUT_LAYERS_QWEN3:
                captures.append(hidden_states.detach().clone())
            del layer, state
            gc.collect()
            torch.cuda.empty_cache()

        if len(captures) != len(OUTPUT_LAYERS_QWEN3):
            raise RuntimeError(
                f"captured {len(captures)} Qwen3 layers, expected {len(OUTPUT_LAYERS_QWEN3)}"
            )
        stacked = torch.stack(captures, dim=1)
        return rearrange(stacked, "b c l d -> b l (c d)")
