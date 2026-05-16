"""Export all FLUX.2 Klein 9B modules to GGUF.

Run from project root:
    uv run python -m models.quantized_klein9b.export_gguf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoConfig

from models.quantized_klein9b.model_sources import (
    FLUX_REPO_ID,
    resolve_model_dirs,
)
from models.quantized_klein9b.quantization import (
    ae_q4_k_m_config,
    klein9b_q4_k_m_config,
)
from models.quantized_qwen3.quantization import qwen3_q4_k_m_config
from torch2vk.quantize import export_q4_k_m_gguf


REPO_ID = FLUX_REPO_ID
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist" / "quantized_klein9b"
DEFAULT_FLUX_GGUF = DEFAULT_OUTPUT_DIR / "flux" / "model.gguf"
DEFAULT_TEXT_ENCODER_GGUF = DEFAULT_OUTPUT_DIR / "text_encoder" / "model.gguf"
DEFAULT_AE_GGUF = DEFAULT_OUTPUT_DIR / "ae" / "model.gguf"


@dataclass(frozen=True, slots=True)
class Klein9BGGUFPaths:
    flux: Path
    text_encoder: Path
    ae: Path


def export_klein9b_q4_k_m_ggufs(
    *,
    model_dir: str | Path | None = None,
    text_encoder_dir: str | Path | None = None,
    ae_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
) -> Klein9BGGUFPaths:
    model_dirs = resolve_model_dirs(
        model_dir=model_dir,
        text_encoder_dir=text_encoder_dir,
        ae_dir=ae_dir,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    flux_path = export_q4_k_m_gguf(
        model_dir=model_dirs.flux,
        output=output / "flux" / "model.gguf",
        config=klein9b_q4_k_m_config(),
        overwrite=overwrite,
    )
    text_config = AutoConfig.from_pretrained(model_dirs.text_encoder)
    text_encoder_path = export_q4_k_m_gguf(
        model_dir=model_dirs.text_encoder,
        output=output / "text_encoder" / "model.gguf",
        config=qwen3_q4_k_m_config(int(text_config.num_hidden_layers)),
        overwrite=overwrite,
    )
    ae_path = export_q4_k_m_gguf(
        model_dir=model_dirs.ae,
        output=output / "ae" / "model.gguf",
        config=ae_q4_k_m_config(),
        overwrite=overwrite,
    )
    return Klein9BGGUFPaths(flux=flux_path, text_encoder=text_encoder_path, ae=ae_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--text-encoder-dir", type=Path)
    parser.add_argument("--ae-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = export_klein9b_q4_k_m_ggufs(
        model_dir=args.model_dir,
        text_encoder_dir=args.text_encoder_dir,
        ae_dir=args.ae_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(paths.flux)
    print(paths.text_encoder)
    print(paths.ae)


if __name__ == "__main__":
    main()
