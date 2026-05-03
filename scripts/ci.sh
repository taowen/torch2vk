#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required but was not found in PATH" >&2
  exit 127
fi

echo "==> ruff"
uv run ruff check src tests

echo "==> pyright"
uv run pyright src tests

echo "==> qwen3 contracts"
uv run python scripts/verify_qwen3_contracts.py

echo "==> qwen3 storage"
uv run python scripts/verify_qwen3_storage.py
uv run python scripts/verify_qwen3_safetensor_weights.py

echo "==> qwen3 shaders"
uv run python scripts/compile_qwen3_shaders.py
uv run python scripts/vulkan_qwen3_embedding_smoke.py
uv run python scripts/vulkan_qwen3_linear_smoke.py
uv run python scripts/vulkan_qwen3_swiglu_smoke.py
uv run python scripts/vulkan_qwen3_argmax_smoke.py
uv run python scripts/vulkan_qwen3_add_smoke.py
uv run python scripts/vulkan_qwen3_rms_norm_smoke.py
uv run python scripts/vulkan_qwen3_rms_norm_rope_smoke.py
uv run python scripts/vulkan_qwen3_set_rows_smoke.py
uv run python scripts/vulkan_qwen3_fa_split_reduce_smoke.py
uv run python scripts/vulkan_qwen3_flash_attn_smoke.py

echo "==> pytest"
set +e
uv run pytest
pytest_status=$?
set -e
if [[ "$pytest_status" -eq 5 ]]; then
  echo "pytest collected no tests; skipping until integration tests are added"
elif [[ "$pytest_status" -ne 0 ]]; then
  exit "$pytest_status"
fi
