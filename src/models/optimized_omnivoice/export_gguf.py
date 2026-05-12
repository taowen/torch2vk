"""Export OmniVoice safetensors to Q4_K_M GGUF."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.hf_cache import load_config_json, resolve_cached_model
from models.optimized_omnivoice.pytorch.example import REPO_ID
from models.optimized_omnivoice.quantization import (
    Q8_TENSOR_NAMES,
    omnivoice_q4_k_m_q6_tensor_names,
)
from torch2vk.quantize import Q4KMQuantizationConfig, export_q4_k_m_gguf


REPO_ROOT = Path(__file__).resolve().parents[3]
QUANTIZE_GGUF_ARCH = "clip"
DEFAULT_Q4_K_M_GGUF = REPO_ROOT / "dist" / "optimized_omnivoice" / "model.gguf"


def export_omnivoice_q4_k_m_gguf(
    *,
    model_dir: str | Path | None = None,
    output: str | Path = DEFAULT_Q4_K_M_GGUF,
    overwrite: bool = False,
) -> Path:
    resolved_model_dir = resolve_cached_model(REPO_ID, model_dir)
    config = load_config_json(resolved_model_dir)
    return export_q4_k_m_gguf(
        model_dir=resolved_model_dir,
        output=output,
        config=Q4KMQuantizationConfig(
            model_name="OmniVoice",
            gguf_arch=QUANTIZE_GGUF_ARCH,
            q6_tensor_names=omnivoice_q4_k_m_q6_tensor_names(_llm_num_hidden_layers(config)),
            q8_tensor_names=Q8_TENSOR_NAMES,
            extra_uint32_metadata=(
                (
                    f"{QUANTIZE_GGUF_ARCH}.audio_vocab_size",
                    _config_int(config, "audio_vocab_size", 1025),
                ),
                (
                    f"{QUANTIZE_GGUF_ARCH}.num_audio_codebook",
                    _config_int(config, "num_audio_codebook", 8),
                ),
                (f"{QUANTIZE_GGUF_ARCH}.audio_mask_id", _config_int(config, "audio_mask_id", 1024)),
            ),
        ),
        overwrite=overwrite,
    )


def _config_int(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int):
        raise TypeError(f"Expected integer config value {key}, got {value!r}")
    return value


def _llm_num_hidden_layers(config: dict[str, object]) -> int:
    llm_config = config.get("llm_config")
    if not isinstance(llm_config, dict):
        raise TypeError(f"Expected dict config value llm_config, got {llm_config!r}")
    value = llm_config.get("num_hidden_layers")
    if not isinstance(value, int):
        raise TypeError(
            f"Expected integer config value llm_config.num_hidden_layers, got {value!r}"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_Q4_K_M_GGUF)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = export_omnivoice_q4_k_m_gguf(
        model_dir=args.model_dir,
        output=args.output,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
