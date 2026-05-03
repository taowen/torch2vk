#!/usr/bin/env python3
"""Verify a real Qwen3 safetensors checkpoint matches torch2vk declarations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from torch2vk.models.qwen3_safetensor.spec import load_qwen3_spec
from torch2vk.models.qwen3_safetensor.weights import verify_qwen3_safetensor_weights


def main() -> int:
    args = _parse_args()
    model_dir = args.model_dir
    if model_dir is None:
        env_model_dir = os.environ.get("QWEN3_SAFETENSOR_DIR")
        if env_model_dir is not None and env_model_dir:
            model_dir = Path(env_model_dir)
    if model_dir is None:
        print("qwen3_safetensor_weights=skip reason=QWEN3_SAFETENSOR_DIR unset")
        return 0

    spec = load_qwen3_spec(model_dir)
    verification = verify_qwen3_safetensor_weights(model_dir, spec=spec)
    verification.raise_for_mismatches()
    print(
        "qwen3_safetensor_weights=ok "
        f"declared={len(verification.manifest.weights)} "
        f"checkpoint_tensors={len(verification.checkpoint_tensors)} "
        f"path={verification.manifest.checkpoint_path}"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Qwen3 safetensors checkpoint dtype and shape contracts.",
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        type=Path,
        help="Directory containing config.json and model.safetensors or index json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
