"""Export Qwen3-ASR safetensors to Q4_K_M GGUF."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.hf_cache import resolve_cached_model
from models.optimized_qwen3_asr.pytorch.example import REPO_ID
from torch2vk.quantize import Q4KMQuantizationConfig, export_q4_k_m_gguf


REPO_ROOT = Path(__file__).resolve().parents[3]
QUANTIZE_GGUF_ARCH = "qwen3-asr"
DEFAULT_Q4_K_M_GGUF = REPO_ROOT / "dist" / "quantized_qwen3_asr" / "model.gguf"


def export_qwen3_asr_q4_k_m_gguf(
    *,
    model_dir: str | Path | None = None,
    output: str | Path = DEFAULT_Q4_K_M_GGUF,
    overwrite: bool = False,
) -> Path:
    return export_q4_k_m_gguf(
        model_dir=resolve_cached_model(REPO_ID, model_dir),
        output=output,
        config=Q4KMQuantizationConfig(
            model_name="Qwen3-ASR",
            gguf_arch=QUANTIZE_GGUF_ARCH,
        ),
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_Q4_K_M_GGUF)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = export_qwen3_asr_q4_k_m_gguf(
        model_dir=args.model_dir,
        output=args.output,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
