#!/usr/bin/env python3
"""Verify the OmniVoice safetensors checkpoint matches torch2vk declarations."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch2vk.models.omnivoice_safetensor.model_directory import resolve_omnivoice_model_dir
from torch2vk.models.omnivoice_safetensor.spec import load_omnivoice_spec
from torch2vk.models.omnivoice_safetensor.weights import verify_omnivoice_safetensor_weights


def main() -> int:
    args = _parse_args()
    model_dir = resolve_omnivoice_model_dir(args.model_dir)
    spec = load_omnivoice_spec(model_dir)
    verification = verify_omnivoice_safetensor_weights(model_dir, spec=spec)
    verification.raise_for_mismatches()
    print(
        "omnivoice_safetensor_weights=ok "
        f"declared={len(verification.manifest.weights)} "
        f"checkpoint_tensors={len(verification.checkpoint_tensors)} "
        f"path={verification.manifest.checkpoint_path}"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify OmniVoice safetensors checkpoint dtype and shape contracts.",
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        type=Path,
        help="Directory containing config.json and model.safetensors.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
