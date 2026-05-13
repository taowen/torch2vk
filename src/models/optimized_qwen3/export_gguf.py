"""Export standalone Qwen3 safetensors to torch2vk Q4_K_M GGUF."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoConfig

from models.hf_cache import resolve_cached_model
from models.optimized_qwen3.quantization import qwen3_q4_k_m_config
from torch2vk.quantize import export_q4_k_m_gguf


REPO_ID = "Qwen/Qwen3-0.6B"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_Q4_K_M_GGUF = REPO_ROOT / "dist" / "optimized_qwen3" / "model.gguf"


def export_qwen3_q4_k_m_gguf(
    *,
    model_dir: str | Path | None = None,
    output: str | Path = DEFAULT_Q4_K_M_GGUF,
    overwrite: bool = False,
) -> Path:
    resolved_model_dir = resolve_cached_model(REPO_ID, model_dir)
    config = AutoConfig.from_pretrained(resolved_model_dir)
    return export_q4_k_m_gguf(
        model_dir=resolved_model_dir,
        output=output,
        config=qwen3_q4_k_m_config(int(config.num_hidden_layers)),
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_Q4_K_M_GGUF)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = export_qwen3_q4_k_m_gguf(
        model_dir=args.model_dir,
        output=args.output,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
