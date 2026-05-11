"""Input preparation for standalone Qwen3 generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_PROMPT = (
    "Explain Vulkan compute shaders in one concise paragraph, focusing on why "
    "they are useful for neural network inference."
)


@dataclass(frozen=True, slots=True)
class PreparedQwen3Inputs:
    prompt: str
    prompt_text: str
    input_ids: np.ndarray

    @property
    def prompt_length(self) -> int:
        return int(self.input_ids.shape[1])


def load_qwen3_tokenizer(model_dir: str | Path) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(Path(model_dir))


def prepare_qwen3_inputs(
    *,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str = DEFAULT_PROMPT,
) -> PreparedQwen3Inputs:
    messages = [{"role": "user", "content": prompt}]
    prompt_text = cast(str, tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ))
    encoded = tokenizer(prompt_text, return_tensors="np")
    input_ids = np.ascontiguousarray(encoded["input_ids"], dtype=np.int64)
    return PreparedQwen3Inputs(
        prompt=prompt,
        prompt_text=prompt_text,
        input_ids=input_ids,
    )
